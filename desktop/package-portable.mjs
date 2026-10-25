import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';

if (process.argv.length !== 5) throw new Error('Usage: node package-portable.mjs TOOLS_DIR APP_FOLDER OUTPUT_DIR');
const [tools, appFolder, output] = process.argv.slice(2).map(value => path.resolve(value));
const source = path.dirname(fileURLToPath(import.meta.url));
const metadata = JSON.parse(fs.readFileSync(path.join(source, 'package.json'), 'utf8'));
const builderPath = path.join(tools, 'node_modules', 'electron-builder');
const builderMetadata = JSON.parse(fs.readFileSync(path.join(builderPath, 'package.json'), 'utf8'));
if (builderMetadata.version !== '26.18.0') throw new Error('Install electron-builder 26.18.0 before building');
for (const file of ['icudtl.dat', 'resources/app.asar', 'resources/backend/XiangQiBackend.exe', 'resources/frontend/index.html']) {
  if (!fs.existsSync(path.join(appFolder, file))) throw new Error('Incomplete application folder: ' + file);
}
fs.mkdirSync(output, { recursive: true });
const licenseFolder = path.join(appFolder, 'licenses', 'build-tools');
fs.mkdirSync(licenseFolder, { recursive: true });
fs.copyFileSync(path.join(builderPath, 'LICENSE'), path.join(licenseFolder, 'electron-builder-LICENSE.txt'));
const builderLib = path.resolve(builderPath, '..', 'app-builder-lib');
const { getMakeNsisPath } = await import(pathToFileURL(path.join(builderLib, 'out', 'toolsets', 'windows.js')));
const compiler = await getMakeNsisPath();
fs.copyFileSync(path.resolve(path.dirname(compiler.path), '..', 'COPYING'), path.join(licenseFolder, 'NSIS-COPYING.txt'));
fs.copyFileSync(path.join(builderLib, 'templates', 'nsis', 'include', 'StdUtils.nsh'), path.join(licenseFolder, 'StdUtils.nsh'));
for (const file of ['portable-NOTICE.txt', 'StdUtils-LGPL-2.1.txt']) {
  fs.copyFileSync(path.join(source, 'licenses', file), path.join(licenseFolder, file));
}
const splash = path.join(output, 'portable-splash.bmp');
const rendered = spawnSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
  '-File', path.join(source, 'portable-splash.ps1'), '-OutputFile', splash], { windowsHide: true, stdio: 'inherit' });
if (rendered.status !== 0) throw new Error('Could not create portable startup screen');
const configuration = JSON.parse(fs.readFileSync(path.join(source, 'portable-builder.json'), 'utf8'));
configuration.directories = { output };
configuration.portable.splashImage = splash;
const { build, Platform, Arch } = await import(pathToFileURL(path.join(builderPath, 'out', 'index.js')));
await build({ projectDir: source, prepackaged: appFolder, publish: 'never',
  targets: Platform.WINDOWS.createTarget('portable', Arch.x64), config: configuration });
const artifact = path.join(output, `XiangQi-Vedio-Reconstruction-${metadata.version}-win-x64-portable.exe`);
if (!fs.existsSync(artifact)) throw new Error('Portable executable was not produced');
console.log('Single-file portable executable: ' + artifact);
