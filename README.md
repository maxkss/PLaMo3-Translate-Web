# PLaMo Translate (On-device)

A Chrome extension that runs [PLaMo 3 Translate](https://huggingface.co/pfnet/plamo-3-translate-2b) (0.8B / 2B) directly in the browser with WebGPU. No text is sent to a server.

- **Selection translation** — select text and click the translate button (or use the context menu, or `Alt+T`). The result streams in.
- **Whole-page translation** — from the popup or `Alt+Shift+T`. Translates the entire page starting from what is on screen, keeps inline formatting (links, bold, code), and follows dynamically added content and SPA navigation. Shows the translation below the original or in place of it; the original can be restored at any time.
- **Image translation** — right-click an image and choose the PLaMo image translation item. Text in the image is read with OCR, translated, erased from the picture and redrawn in the same position, size and color.
- 53 target languages. The default is Korean.

| Model | Download | Converted weights |
|---|---|---|
| 0.8B (default) | 0.50 GB | [chiro0318/plamo-3-translate-0.8b-onnx-webgpu](https://huggingface.co/chiro0318/plamo-3-translate-0.8b-onnx-webgpu) |
| 2B | 1.62 GB | [chiro0318/plamo-3-translate-2b-onnx-webgpu](https://huggingface.co/chiro0318/plamo-3-translate-2b-onnx-webgpu) |

The models are 4-bit quantized ONNX conversions that run on onnxruntime-web. Without WebGPU the extension falls back to WebAssembly on the CPU, which is very slow.

## Installation

1. Download `plamo-translate-<version>.zip` from the [latest release](https://github.com/maxkss/PLaMo3-Translate-Web/releases/latest) and unzip it, or build the extension yourself (see [Build](#build)).
2. Open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked** and select the unzipped folder (or `extension/dist` if you built it).
3. Open the extension's options page (the model settings link in the popup), choose 0.8B or 2B and click the download button.

The model is downloaded from Hugging Face at a pinned commit, and every file is checked against its size and SHA-256. An interrupted download resumes where it stopped. Models are stored in the extension's Origin Private File System, so translation works offline afterwards. Both models can be installed side by side and switched at any time.

The OCR model for image translation (PP-OCRv5, 21 MB) and the inpainting model (MI-GAN, 28 MB) are downloaded the first time they are needed.

## Build

Requires Node.js and npm.

```bash
cd extension
npm install
npm run build
```

The extension is written to `extension/dist`. `npm run watch` rebuilds on change, `npm test` runs the unit tests and `npm run typecheck` checks types.

### Converting the models (optional)

The published weights can be reproduced from the original PFN models. This needs [uv](https://docs.astral.sh/uv/) and about 15 GB of free disk space.

```bash
tools/convert/convert.sh 0.8b   # or 2b
```

The result is `models/onnx/plamo-3-translate-<size>/onnx/model_q4f16.onnx` and `model_q4f16.onnx_data`. To install a local conversion, serve it with `npm run serve-model` in `extension/` and enter the URL under the advanced settings on the options page, or import the folder there. Local files must match the checksums in `extension/src/shared/common.ts`.

## License

The source code in this repository is released under the [MIT License](LICENSE).

The models are licensed separately. The PLaMo 3 Translate models and the converted weights are covered by the [PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-ja). Personal and non-commercial use is free; commercial use requires prior registration with Preferred Networks or a separate agreement. The license also applies when the converted models are redistributed. Built with PLaMo.

Third-party models downloaded by the extension:

- [PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR) (OCR) and its character list in `extension/public/ocr/ppocrv5_dict.json` — Apache-2.0, PaddlePaddle.
- [MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN) (inpainting) — MIT, Picsart AI Research.
