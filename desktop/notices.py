import importlib.metadata
from pathlib import Path
import shutil
import sys
import zipfile

destination, source, build_root = map(Path, sys.argv[1:])
notices = destination / 'licenses'
notices.mkdir(exist_ok=True)
shutil.copy2(source / 'THIRD_PARTY.md', notices)
(notices / 'docs').mkdir(exist_ok=True)
shutil.copy2(source / 'docs' / 'models.md', notices / 'docs' / 'models.md')
shutil.copy2(source / 'desktop' / 'README.md', destination / '使用说明.md')
packages = []
for distribution in importlib.metadata.distributions():
    name = distribution.metadata['Name']
    packages.append(f'{name}=={distribution.version}')
    for item in distribution.files or []:
        if any(part.lower().startswith(('license', 'copying', 'notice')) for part in item.parts):
            original = Path(distribution.locate_file(item))
            if original.is_file():
                target = notices / 'python' / name / Path(*item.parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
(notices / 'python-packages.txt').write_text('\n'.join(sorted(packages)) + '\n', encoding='utf-8')
for name in ('react', 'react-dom', 'scheduler', 'lucide-react'):
    for item in (build_root / 'frontend' / 'node_modules' / name).glob('LICENSE*'):
        target = notices / 'frontend' / name / item.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
with zipfile.ZipFile(notices / 'application-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for folder in ('backend', 'desktop', 'frontend', 'docs', 'scripts'):
        for item in (source / folder).rglob('*'):
            if item.is_file() and not any(part in ('node_modules', '__pycache__', '.pytest_cache', 'dist', 'data') for part in item.relative_to(source).parts):
                archive.write(item, item.relative_to(source).as_posix())
    for name in ('README.md', 'THIRD_PARTY.md', '.env.example', 'pytest.ini'):
        archive.write(source / name, name)
