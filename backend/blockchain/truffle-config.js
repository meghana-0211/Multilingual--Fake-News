module.exports = {
  networks: {
    development: {
      host: "127.0.0.1",
      port: 8545,
      network_id: 1337,  // Match Ganache!
      gas: 6721975,
      gasPrice: 20000000000
    }
  },
  
  compilers: {
    solc: {
      version: "0.8.19"
    }
  }
};