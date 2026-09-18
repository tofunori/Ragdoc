#!/usr/bin/env python3
"""Bundle pinned bootstrap tooling, immutable engine sources and the Claude extension."""
from pathlib import Path
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
UV_VERSION = "0.10.0"
UV_SHA256 = "82d4b99dc6ea686695b5ee142ceba03dd3e3eda2b414e94215ab7bce94972fbb"
UV_URL = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-aarch64-apple-darwin.tar.gz"


def package(resources: Path):
    if subprocess.check_output(['uname', '-m'], text=True).strip() != 'arm64':
        raise SystemExit('The local runtime package currently targets Apple silicon.')
    engine = resources / 'LocalEngine'
    engine.mkdir(parents=True, exist_ok=True)
    paths = ['pyproject.toml', 'uv.lock', 'LICENSE', 'README.md']
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'src').rglob('*.py')) if '__pycache__' not in p.parts]
    paths += ['scripts/'+name for name in ['ragdrop_local.py', 'index_incremental.py', 'index_artifacts.py']]
    identity = hashlib.sha256()
    for relative in paths:
        data = (ROOT/relative).read_bytes()
        identity.update(relative.encode()); identity.update(b'\0'); identity.update(data)
        dest = engine/relative; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data)
    (engine/'engine-id').write_text(identity.hexdigest()+'\n')
    cache = ROOT/'Ragdrop/.build/bootstrap'/f'uv-{UV_VERSION}.tar.gz'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        data = urllib.request.urlopen(UV_URL, timeout=90).read()
        if hashlib.sha256(data).hexdigest() != UV_SHA256: raise SystemExit('uv checksum mismatch')
        cache.write_bytes(data)
    data = cache.read_bytes()
    if hashlib.sha256(data).hexdigest() != UV_SHA256: raise SystemExit('Cached uv checksum mismatch')
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        member = archive.getmember('uv-aarch64-apple-darwin/uv')
        (resources/'uv').write_bytes(archive.extractfile(member).read())
    (resources/'uv').chmod(0o755)
    # Sign nested executable before the containing app is signed.
    subprocess.run(['/usr/bin/codesign','--force','--sign','-',str(resources/'uv')],check=True)
    with zipfile.ZipFile(resources/'Ragdoc.mcpb','w',zipfile.ZIP_DEFLATED) as archive:
        for relative in ['manifest.json','server/index.js']:
            archive.write(ROOT/'Ragdrop/Integrations/ClaudeDesktop'/relative,relative)
        archive.write(ROOT/'LICENSE','LICENSE')
    licenses = resources/'ThirdParty'; licenses.mkdir(exist_ok=True)
    # uv is dual MIT/Apache-2.0; retain both licenses with the bundled executable.
    for name in ['LICENSE-MIT','LICENSE-APACHE']:
        url=f'https://raw.githubusercontent.com/astral-sh/uv/{UV_VERSION}/{name}'
        (licenses/('uv-'+name)).write_bytes(urllib.request.urlopen(url,timeout=30).read())
    (licenses/'README.txt').write_text('uv '+UV_VERSION+' from '+UV_URL+'\nArchive SHA-256: '+UV_SHA256+'\nPython and locked Python dependencies are downloaded during setup; their licenses remain in the managed environment.\n')

if __name__ == '__main__':
    package(Path(sys.argv[1]).resolve())
