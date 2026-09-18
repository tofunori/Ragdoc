const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { loadConnection } = require('./index.js');

test('connection requires setup and stays inside its managed engine', () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'ragdoc-extension-'));
  try {
    assert.throws(() => loadConnection(home), /prepare a library/);
    const support = path.join(home, 'Library/Application Support/Ragdrop');
    const engine = path.join(support, 'Engines/test');
    const python = path.join(engine, '.venv/bin/python');
    const script = path.join(engine, 'scripts/ragdrop_local.py');
    fs.mkdirSync(path.dirname(python), { recursive: true });
    fs.mkdirSync(path.dirname(script), { recursive: true });
    fs.writeFileSync(python, '', { mode: 0o755 }); fs.writeFileSync(script, '');
    const connection = { version:1, python, script, library:path.join(home, 'Papers with spaces') };
    fs.writeFileSync(path.join(support, 'local-connection.json'), JSON.stringify(connection));
    assert.deepEqual(loadConnection(home), connection);
    connection.python = '/usr/bin/python3';
    fs.writeFileSync(path.join(support, 'local-connection.json'), JSON.stringify(connection));
    assert.throws(() => loadConnection(home), /Ragdrop engine/);
  } finally { fs.rmSync(home, { recursive: true, force: true }); }
});
