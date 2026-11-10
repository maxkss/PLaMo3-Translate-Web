/** Stylesheets imported by the content script are inlined as text (build.mjs: loader ".css": "text"). */
declare module "*.css" {
  const css: string;
  export default css;
}
