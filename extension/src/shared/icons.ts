/** Inline 24×24 stroke icons (no icon library). */

const PATHS = {
  translate: `<path d="m5 8 6 6M4 14l6-6 2-3M2 5h12M7 2h1M22 22l-5-10-5 10M14 18h6"/>`,
  settings: `<path d="M21 4h-7M10 4H3M21 12h-9M8 12H3M21 20h-5M12 20H3M14 2v4M8 10v4M16 18v4"/>`,
  x: `<path d="M18 6 6 18M6 6l12 12"/>`,
  copy: `<rect x="8" y="8" width="14" height="14" rx="2"/><path d="M4 16a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2"/>`,
  check: `<path d="M20 6 9 17l-5-5"/>`,
  warning: `<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4M12 17h.01"/>`,
  error: `<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6M9 9l6 6"/>`,
  eye: `<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>`,
} as const;

export type IconName = keyof typeof PATHS;

export const icon = (name: IconName, size = 16): string =>
  `<svg class="i" viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${PATHS[name]}</svg>`;

/** Replace every `<i data-icon="name">` placeholder in the document with its SVG. */
export function hydrateIcons(root: ParentNode = document): void {
  for (const el of root.querySelectorAll<HTMLElement>("[data-icon]")) el.outerHTML = icon(el.dataset.icon as IconName);
}
