// blockchain/contracts/PublisherRegistry.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract PublisherRegistry {
    struct Publisher {
        string name;
        address walletAddress;
        uint256 reputationScore;
        uint256 registrationTime;
        bool isActive;
    }
    
    mapping(address => Publisher) public publishers;
    address[] public publisherAddresses;
    
    address public admin;
    
    event PublisherRegistered(address indexed publisher, string name);
    event PublisherDeactivated(address indexed publisher);
    event ReputationUpdated(address indexed publisher, uint256 newScore);
    
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin");
        _;
    }
    
    constructor() {
        admin = msg.sender;
    }
    
    function registerPublisher(address _publisher, string memory _name) public onlyAdmin {
        require(!publishers[_publisher].isActive, "Already registered");
        
        publishers[_publisher] = Publisher({
            name: _name,
            walletAddress: _publisher,
            reputationScore: 100,
            registrationTime: block.timestamp,
            isActive: true
        });
        
        publisherAddresses.push(_publisher);
        emit PublisherRegistered(_publisher, _name);
    }
    
    function updateReputation(address _publisher, uint256 _newScore) public onlyAdmin {
        require(publishers[_publisher].isActive, "Publisher not active");
        publishers[_publisher].reputationScore = _newScore;
        emit ReputationUpdated(_publisher, _newScore);
    }
    
    function deactivatePublisher(address _publisher) public onlyAdmin {
        publishers[_publisher].isActive = false;
        emit PublisherDeactivated(_publisher);
    }
    
    function isRegistered(address _publisher) public view returns (bool) {
        return publishers[_publisher].isActive;
    }
    
    function getPublisher(address _publisher) public view returns (Publisher memory) {
        return publishers[_publisher];
    }
}