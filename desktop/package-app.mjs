import path from 'node:path';
import fs from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';

const [buildRoot, outputRoot] = process.argv.slice(2).map(value => path.resolve(value));
if (!buildRoot || !outputRoot) throw new Error('Usage: node package-app.mjs BUILD_ROOT OUTPUT_ROOT');
const source = path.dirname(fileURLToPath(import.meta.url));
const { packager } = await import(pathToFileURL(path.join(buildRoot, 'tools/node_modules/@electron/packager/dist/index.js')));
const options = {
  dir: source, name: 'XiangQi Vedio Reconstruction', appVersion: '0.2.0',
  electronVersion: '44.9.0', platform: 'win32', arch: 'x64', asar: true,
  out: path.join(buildRoot, 'packaged'), overwrite: true,
  tmpdir: path.join(buildRoot, 'packager-tmp'),
  download: { cacheRoot: path.join(buildRoot, 'cache/electron') },
  extraResource: [path.join(buildRoot, 'backend'), path.join(buildRoot, 'frontend/dist')],
  ignore: [/backend\.spec$/, /backend_entry\.py$/, /notices\.py$/, /package-app\.mjs$/, /build\.ps1$/],
  win32metadata: { ProductName: 'XiangQi Vedio Reconstruction', FileDescription: 'XiangQi Vedio Reconstruction' },
  afterCopyExtraResources: [async ({ buildPath }) => {
    const resources = path.join(buildPath, 'resources');
    fs.renameSync(path.join(resources, 'dist'), path.join(resources, 'frontend'));
  }],
};
if (process.env.ELECTRON_ZIP_DIR) options.electronZipDir = process.env.ELECTRON_ZIP_DIR;
const [folder] = await packager(options);
const destination = path.join(outputRoot, 'XiangQi Vedio Reconstruction');
if (fs.existsSync(destination)) throw new Error('Output folder already exists: ' + destination);
fs.renameSync(folder, destination);
console.log(destination);
