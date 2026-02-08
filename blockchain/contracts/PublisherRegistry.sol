// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * PUBLISHER REGISTRY CONTRACT
 * 
 * Think: Database of verified publishers (like Twitter verified checkmarks)
 * 
 * What it stores:
 * - Publisher name (e.g., "BBC News")
 * - Wallet address (their blockchain ID)
 * - Reputation score (0-100)
 * - Registration date
 * - Active status
 * 
 * Who can use it:
 * - Admin: Can add/remove publishers
 * - Anyone: Can check if publisher is verified
 */

contract PublisherRegistry {
    
    // ============================================
    // DATA STRUCTURES
    // ============================================
    
    /**
     * Publisher struct - Think: Database row for each publisher
     */
    struct Publisher {
        string name;              // "BBC News"
        address walletAddress;    // 0x742d35Cc...
        uint256 reputationScore;  // 0 to 100
        uint256 registrationTime; // Unix timestamp
        bool isActive;            // true/false
    }
    
    // Mapping = Dictionary in Python
    // Key: address → Value: Publisher struct
    mapping(address => Publisher) public publishers;
    
    // Array of all publisher addresses (for iteration)
    address[] public publisherAddresses;
    
    // Admin address (only admin can add publishers)
    address public admin;
    
    // ============================================
    // EVENTS (Like console.log but on blockchain)
    // ============================================
    
    event PublisherRegistered(address indexed publisher, string name);
    event PublisherDeactivated(address indexed publisher);
    event ReputationUpdated(address indexed publisher, uint256 newScore);
    
    // ============================================
    // MODIFIERS (Access Control)
    // ============================================
    

    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can do this");
        _;
    }
    
    // ============================================
    // CONSTRUCTOR (Runs once when deployed)
    // ============================================
    
    constructor() {
        // Person who deploys contract becomes admin
        admin = msg.sender;
    }
    
    // ============================================
    // FUNCTIONS
    // ============================================
    
    /**
     * Register a new publisher
     * 
     * @param _publisher Address of the publisher
     * @param _name Name of the publisher (e.g., "BBC")
     * 
     * Only admin can call this
     */
    function registerPublisher(address _publisher, string memory _name) 
        public 
        onlyAdmin 
    {
        // Check publisher isn't already registered
        require(!publishers[_publisher].isActive, "Already registered");
        
        // Create new publisher
        publishers[_publisher] = Publisher({
            name: _name,
            walletAddress: _publisher,
            reputationScore: 100,  // Start with perfect score
            registrationTime: block.timestamp,  // Current time
            isActive: true
        });
        
        // Add to array
        publisherAddresses.push(_publisher);
        
        // Emit event (like logging)
        emit PublisherRegistered(_publisher, _name);
    }
    
    /**
     * Update publisher reputation
     * 
     * @param _publisher Publisher address
     * @param _newScore New reputation score (0-100)
     */
    function updateReputation(address _publisher, uint256 _newScore) 
        public 
        onlyAdmin 
    {
        require(publishers[_publisher].isActive, "Publisher not found");
        require(_newScore <= 100, "Score must be 0-100");
        
        publishers[_publisher].reputationScore = _newScore;
        emit ReputationUpdated(_publisher, _newScore);
    }
    
    /**
     * Deactivate a publisher (like banning)
     * 
     * @param _publisher Publisher address
     */
    function deactivatePublisher(address _publisher) 
        public 
        onlyAdmin 
    {
        publishers[_publisher].isActive = false;
        emit PublisherDeactivated(_publisher);
    }
    
    /**
     * Check if address is a verified publisher
     * 
     * @param _publisher Address to check
     * @return bool True if verified
     * 
     * Anyone can call this (public read)
     */
    function isVerified(address _publisher) 
        public 
        view 
        returns (bool) 
    {
        return publishers[_publisher].isActive;
    }
    
    /**
     * Get publisher details
     * 
     * @param _publisher Publisher address
     * @return Publisher struct
     */
    function getPublisher(address _publisher) 
        public 
        view 
        returns (Publisher memory) 
    {
        return publishers[_publisher];
    }
    
    /**
     * Get total number of publishers
     */
    function getPublisherCount() 
        public 
        view 
        returns (uint256) 
    {
        return publisherAddresses.length;
    }
}
