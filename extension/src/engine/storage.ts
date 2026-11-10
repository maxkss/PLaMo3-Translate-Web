/**
 * Model file storage in the extension's Origin Private File System (OPFS).
 *
 * Each model lives in /<ModelInfo.dir>/<basename>. `installed.json` records the files whose size and
 * SHA-256 matched the catalog (shared/common.ts); only those count as installed.
 *
 * Downloads stream straight to disk while hashing, resume an interrupted file with an HTTP Range
 * request, and check the storage quota before starting.
 */

import { MODELS, type ModelFile, type ModelId, type ModelInfo, type StoredModel } from "../shared/common.ts";
import { Sha256 } from "./sha256.ts";

const MANIFEST = "installed.json";

interface Manifest {
  version: 2;
  files: Record<string, { size: number; sha256: string }>;
}

export function basename(p: string): string {
  return p.split("/").pop()!;
}

/** File name inside the model's OPFS directory. */
export const localName = (f: ModelFile) => f.name ?? basename(f.path);

async function modelDir(m: StoredModel, create: boolean): Promise<FileSystemDirectoryHandle | null> {
  const root = await navigator.storage.getDirectory();
  try {
    return await root.getDirectoryHandle(m.dir, { create });
  } catch {
    return null;
  }
}

async function readManifest(dir: FileSystemDirectoryHandle): Promise<Manifest> {
  try {
    const m = JSON.parse(await (await (await dir.getFileHandle(MANIFEST)).getFile()).text());
    if (m?.version === 2) return m as Manifest;
  } catch {
    /* missing or old format */
  }
  return { version: 2, files: {} };
}

async function writeManifest(dir: FileSystemDirectoryHandle, man: Manifest): Promise<void> {
  const w = await (await dir.getFileHandle(MANIFEST, { create: true })).createWritable();
  await w.write(JSON.stringify(man));
  await w.close();
}

async function fileSize(dir: FileSystemDirectoryHandle, name: string): Promise<number> {
  try {
    return (await (await dir.getFileHandle(name)).getFile()).size;
  } catch {
    return -1;
  }
}

const verified = (man: Manifest, f: ModelFile) => {
  const e = man.files[localName(f)];
  return !!e && e.size === f.size && e.sha256 === f.sha256;
};

export async function isInstalled(m: StoredModel): Promise<boolean> {
  const dir = await modelDir(m, false);
  if (!dir) return false;
  const man = await readManifest(dir);
  for (const f of m.files) {
    if (!verified(man, f) || (await fileSize(dir, localName(f))) !== f.size) return false;
  }
  return true;
}

export async function listInstalled(): Promise<ModelId[]> {
  const out: ModelId[] = [];
  for (const m of Object.values(MODELS)) if (await isInstalled(m)) out.push(m.id);
  return out;
}

export async function getModelFiles(m: ModelInfo): Promise<{ model: File; data: File }> {
  const dir = await modelDir(m, false);
  if (!dir) throw new Error("model not installed");
  const [model, data] = await Promise.all(m.files.map(async (f) => (await dir.getFileHandle(basename(f.path))).getFile()));
  return { model, data };
}

/** All of the model's files, in catalog order. */
export async function getStoredFiles(m: StoredModel): Promise<File[]> {
  const dir = await modelDir(m, false);
  if (!dir) throw new Error("model not installed");
  return Promise.all(m.files.map(async (f) => (await dir.getFileHandle(localName(f))).getFile()));
}

export async function deleteModel(m: StoredModel): Promise<void> {
  const root = await navigator.storage.getDirectory();
  await root.removeEntry(m.dir, { recursive: true }).catch(() => {});
}

export type Progress = (loaded: number, total: number) => void;

class ChecksumError extends Error {}

/** Hash an existing (partial) file so a resumed download can continue the same hash. */
async function hashExisting(file: File, h: Sha256, onBytes: (n: number) => void, signal?: AbortSignal): Promise<void> {
  const reader = file.stream().getReader();
  for (;;) {
    signal?.throwIfAborted();
    const { done, value } = await reader.read();
    if (done) return;
    h.update(value);
    onBytes(value.byteLength);
  }
}

async function ensureQuota(need: number): Promise<void> {
  if (need <= 0) return;
  const { quota = Infinity, usage = 0 } = await navigator.storage.estimate();
  if (quota - usage < need * 1.05) {
    throw new Error(`저장 공간이 부족합니다 (필요 ${(need / 1e9).toFixed(2)} GB, 남음 ${((quota - usage) / 1e9).toFixed(2)} GB)`);
  }
}

/**
 * Download the model's files from `baseUrl` (default: the pinned Hugging Face commit) into OPFS.
 * Already-verified files are skipped; a partially written file is resumed with a Range request.
 */
export async function downloadModel(m: StoredModel, baseUrl: string, onProgress: Progress, signal?: AbortSignal): Promise<void> {
  const dir = (await modelDir(m, true))!;
  const man = await readManifest(dir);
  const total = m.files.reduce((a, f) => a + f.size, 0);
  let done = 0;
  const report = (n: number) => onProgress((done += n), total);

  // files already verified need no work; everything else needs at most its full size of new space
  let need = 0;
  for (const f of m.files) if (!verified(man, f)) need += f.size - Math.max(0, await fileSize(dir, localName(f)));
  await ensureQuota(need);

  for (const f of m.files) {
    const name = localName(f);
    if (verified(man, f) && (await fileSize(dir, name)) === f.size) {
      report(f.size);
      continue;
    }
    delete man.files[name];
    await writeManifest(dir, man);

    // up to two attempts: a resumed file that fails the checksum is retried from scratch once
    for (let attempt = 0; ; attempt++) {
      try {
        await downloadFile(dir, f, f.url ?? new URL(f.path, baseUrl).href, attempt > 0, report, signal);
        break;
      } catch (e) {
        if (e instanceof ChecksumError && attempt === 0) {
          done -= f.size;
          continue;
        }
        throw e;
      }
    }
    man.files[name] = { size: f.size, sha256: f.sha256 };
    await writeManifest(dir, man);
  }
}

async function downloadFile(
  dir: FileSystemDirectoryHandle,
  f: ModelFile,
  url: string,
  fresh: boolean,
  report: (n: number) => void,
  signal?: AbortSignal,
): Promise<void> {
  const name = localName(f);
  const handle = await dir.getFileHandle(name, { create: true });
  let existing = fresh ? null : await handle.getFile();
  if (existing && existing.size > f.size) existing = null;

  let hasher = new Sha256();
  let offset = 0;
  if (existing && existing.size > 0) {
    await hashExisting(existing, hasher, report, signal);
    offset = existing.size;
  }

  let res: Response | null = null;
  if (offset < f.size) {
    res = await fetch(url, { signal, cache: "no-store", headers: offset ? { Range: `bytes=${offset}-` } : {} });
    if (res.status === 200 && offset) {
      // server ignored the Range header: start over
      report(-offset);
      offset = 0;
      hasher = new Sha256();
    } else if (!res.ok || (offset && res.status !== 206)) {
      throw new Error(`${f.path}: HTTP ${res.status}`);
    }
  }

  const w = await handle.createWritable({ keepExistingData: offset > 0 });
  try {
    if (offset) await w.seek(offset);
    else await w.truncate(0);
    if (res?.body) {
      const reader = res.body.getReader();
      // batch small network chunks into ~4 MB writes
      const BATCH = 4 << 20;
      let pending: Uint8Array[] = [];
      let pendingBytes = 0;
      const flush = async () => {
        if (!pendingBytes) return;
        const buf = new Uint8Array(pendingBytes);
        let o = 0;
        for (const c of pending) buf.set(c, (o += c.byteLength) - c.byteLength);
        pending = [];
        pendingBytes = 0;
        await w.write(buf);
      };
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (offset + value.byteLength > f.size) throw new Error(`${f.path}: larger than expected`);
        hasher.update(value);
        offset += value.byteLength;
        pending.push(value);
        pendingBytes += value.byteLength;
        if (pendingBytes >= BATCH) await flush();
        report(value.byteLength);
      }
      await flush();
    }
    await w.close();
  } catch (e) {
    // keep what was written so the next attempt can resume
    await w.close().catch(() => w.abort().catch(() => {}));
    throw e;
  }
  if (offset !== f.size) throw new Error(`${f.path}: 크기가 다릅니다 (${offset} / ${f.size})`);
  const got = hasher.hex();
  if (got !== f.sha256) {
    await dir.removeEntry(name).catch(() => {});
    throw new ChecksumError(`${f.path}: 체크섬이 일치하지 않습니다`);
  }
}
