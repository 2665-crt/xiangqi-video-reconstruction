const net = require('node:net');
const path = require('node:path');
const { createHash } = require('node:crypto');

async function claimInstance(dataRoot, recordingRoot, isDataBusy = async () => false) {
  dataRoot = path.resolve(dataRoot);
  recordingRoot = path.resolve(recordingRoot);
  const identity = createHash('sha256').update(dataRoot.toLowerCase()).digest('hex').slice(0, 24);
  for (let id = 1; ; id++) {
    const server = net.createServer(socket => socket.end());
    const address = `\\\\.\\pipe\\xiangqi-${identity}-${id}`;
    const claimed = await new Promise((resolve, reject) => {
      server.once('error', error => error.code === 'EADDRINUSE' ? resolve(false) : reject(error));
      server.listen(address, () => resolve(true));
    });
    if (!claimed) continue;
    const instanceData = id === 1 ? dataRoot : path.join(dataRoot, 'instances', String(id));
    let backendAlive;
    try {
      backendAlive = await isDataBusy(instanceData);
    } catch (error) {
      await new Promise(resolve => server.close(resolve));
      throw error;
    }
    if (backendAlive) {
      await new Promise(resolve => server.close(resolve));
      continue;
    }
    let released = false;
    return {
      id,
      dataRoot: instanceData,
      recordingRoot: id === 1 ? recordingRoot : path.join(recordingRoot, 'instances', String(id)),
      release: () => new Promise(resolve => {
        if (released) return resolve();
        released = true;
        server.close(resolve);
      }),
    };
  }
}

module.exports = { claimInstance };
