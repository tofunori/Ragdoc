// Claude supplies Node. Ragdrop supplies the engine; no secrets are stored here.
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');

function loadConnection(home = os.homedir()) {
  const support = path.join(home, 'Library', 'Application Support', 'Ragdrop');
  let connection;
  try {
    connection = JSON.parse(fs.readFileSync(path.join(support, 'local-connection.json'), 'utf8'));
  } catch {
    throw new Error('Open Ragdrop and prepare a library on This Mac before connecting Claude.');
  }
  const { version, python, script, library } = connection;
  if (version !== 1 || ![python, script, library].every(value => typeof value === 'string' && path.isAbsolute(value))) {
    throw new Error('The local connection is incomplete. Repair the engine in Ragdrop Settings.');
  }
  const engineRoot = path.join(support, 'Engines') + path.sep;
  if (!python.startsWith(engineRoot) || !script.startsWith(engineRoot) || path.dirname(path.dirname(python)) !== path.join(path.dirname(path.dirname(script)), '.venv')) {
    throw new Error('The local connection does not point to a Ragdrop engine. Run setup again.');
  }
  fs.accessSync(python, fs.constants.X_OK);
  fs.accessSync(script, fs.constants.R_OK);
  return connection;
}

function main() {
  try {
    const connection = loadConnection();
    const env = { ...process.env, PYTHON_DOTENV_DISABLED: '1' };
    delete env.PYTHONPATH; delete env.PYTHONHOME; delete env.VIRTUAL_ENV;
    const child = spawn(connection.python, [connection.script, '--root', connection.library, 'mcp'], {
      stdio: 'inherit', env, cwd: path.dirname(path.dirname(connection.script)), shell: false
    });
    child.on('error', () => { console.error('Ragdoc could not start. Repair the local engine in Ragdrop Settings.'); process.exitCode = 1; });
    child.on('exit', (code, signal) => { process.exitCode = code ?? (signal ? 1 : 0); });
    for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal));
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
if (require.main === module) main();
module.exports = { loadConnection };
