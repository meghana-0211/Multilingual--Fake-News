// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * ANNOTATION REGISTRY CONTRACT
 * 
 * Think: Comment section that can't be deleted
 * 
 * What it does:
 * - Fact-checkers can flag articles
 * - Add corrections/notes
 * - All permanently stored
 * - Can't be removed or edited
 * 
 * Why this matters:
 * - Transparent fact-checking
 * - Can't hide corrections
 * - History is preserved
 */

contract AnnotationRegistry {
    
    // ============================================
    // DATA STRUCTURES
    // ============================================
    
    /**
     * Flag types for annotations
     */
    enum FlagType {
        MISLEADING,  // 0 - Partially false
        FALSE,       // 1 - Completely false
        SATIRE,      // 2 - Joke/satire
        UNVERIFIED,  // 3 - Can't verify
        CORRECT      // 4 - Actually correct (dispute fake label)
    }
    
    /**
     * Annotation struct
     * Think: One correction/flag on an article
     */
    struct Annotation {
        bytes32 articleHash;      // Which article?
        address factChecker;      // Who flagged it?
        FlagType flagType;        // What's wrong?
        string ipfsHash;          // Link to detailed explanation
        uint256 timestamp;        // When flagged?
        uint256 confidence;       // How sure? (0-100)
    }
    
    // Mapping: articleHash → array of Annotations
    // One article can have multiple fact-checks
    mapping(bytes32 => Annotation[]) public articleAnnotations;
    
    // Who is a verified fact-checker?
    mapping(address => bool) public verifiedFactCheckers;
    
    // Admin (can verify fact-checkers)
    address public admin;
    
    // ============================================
    // EVENTS
    // ============================================
    
    event AnnotationAdded(
        bytes32 indexed articleHash,
        address indexed factChecker,
        FlagType flagType,
        uint256 confidence
    );
    
    event FactCheckerVerified(address indexed factChecker);
    event FactCheckerRemoved(address indexed factChecker);
    
    // ============================================
    // MODIFIERS
    // ============================================
    
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin");
        _;
    }
    
    modifier onlyVerifiedFactChecker() {
        require(
            verifiedFactCheckers[msg.sender],
            "You must be a verified fact-checker"
        );
        _;
    }
    
    // ============================================
    // CONSTRUCTOR
    // ============================================
    
    constructor() {
        admin = msg.sender;
    }
    
    // ============================================
    // ADMIN FUNCTIONS
    // ============================================
    
    /**
     * Verify a fact-checker
     * 
     * @param _factChecker Address to verify
     * 
     * Only admin can verify fact-checkers
     * Think: Giving someone "fact-checker badge"
     */
    function verifyFactChecker(address _factChecker) 
        public 
        onlyAdmin 
    {
        verifiedFactCheckers[_factChecker] = true;
        emit FactCheckerVerified(_factChecker);
    }
    
    /**
     * Remove fact-checker verification
     */
    function removeFactChecker(address _factChecker) 
        public 
        onlyAdmin 
    {
        verifiedFactCheckers[_factChecker] = false;
        emit FactCheckerRemoved(_factChecker);
    }
    
    // ============================================
    // FACT-CHECKER FUNCTIONS
    // ============================================
    
    /**
     * Add an annotation to an article
     * 
     * Flow:
     * 1. Fact-checker analyzes article
     * 2. Writes detailed correction (stored in IPFS)
     * 3. Calls this function with flag type + IPFS link
     * 4. Annotation permanently stored on blockchain
     * 
     * @param _articleHash Hash of article being flagged
     * @param _flagType Type of issue (MISLEADING/FALSE/etc)
     * @param _ipfsHash Link to detailed correction (IPFS)
     * @param _confidence How confident? (0-100)
     * 
     * Why IPFS?
     * - Blockchain storage is expensive
     * - IPFS = decentralized file storage
     * - Store detailed correction on IPFS
     * - Store only IPFS link on blockchain
     * 
     * Example:
     * 1. Write correction: "This is false because..."
     * 2. Upload to IPFS → get hash: "Qm..."
     * 3. Call: addAnnotation(articleHash, FALSE, "Qm...", 95)
     */
    function addAnnotation(
        bytes32 _articleHash,
        FlagType _flagType,
        string memory _ipfsHash,
        uint256 _confidence
    ) 
        public 
        onlyVerifiedFactChecker 
    {
        require(_confidence <= 100, "Confidence must be 0-100");
        
        // Create annotation
        Annotation memory newAnnotation = Annotation({
            articleHash: _articleHash,
            factChecker: msg.sender,
            flagType: _flagType,
            ipfsHash: _ipfsHash,
            timestamp: block.timestamp,
            confidence: _confidence
        });
        
        // Add to array for this article
        articleAnnotations[_articleHash].push(newAnnotation);
        
        // Emit event
        emit AnnotationAdded(
            _articleHash,
            msg.sender,
            _flagType,
            _confidence
        );
    }
    
    // ============================================
    // PUBLIC READ FUNCTIONS
    // ============================================
    
    /**
     * Get all annotations for an article
     * 
     * @param _articleHash Article hash
     * @return Annotation[] Array of all annotations
     * 
     * Anyone can call this - transparency!
     * 
     * Example:
     * annotations = contract.getAnnotations(articleHash)
     * for ann in annotations:
     *     print(f"{ann.factChecker} flagged as {ann.flagType}")
     *     print(f"Details: ipfs.io/ipfs/{ann.ipfsHash}")
     */
    function getAnnotations(bytes32 _articleHash) 
        public 
        view 
        returns (Annotation[] memory) 
    {
        return articleAnnotations[_articleHash];
    }
    
    /**
     * Get number of annotations for article
     */
    function getAnnotationCount(bytes32 _articleHash) 
        public 
        view 
        returns (uint256) 
    {
        return articleAnnotations[_articleHash].length;
    }
    
    /**
     * Check if article has been flagged
     * 
     * @param _articleHash Article hash
     * @return bool True if has any annotations
     */
    function hasBeenFlagged(bytes32 _articleHash) 
        public 
        view 
        returns (bool) 
    {
        return articleAnnotations[_articleHash].length > 0;
    }
    
    /**
     * Get consensus on article
     * Returns most common flag type among fact-checkers
     * 
     * @param _articleHash Article hash
     * @return FlagType Most common flag
     * @return uint256 Count of that flag
     */
    function getConsensus(bytes32 _articleHash) 
        public 
        view 
        returns (FlagType, uint256) 
    {
        Annotation[] memory annotations = articleAnnotations[_articleHash];
        
        if (annotations.length == 0) {
            return (FlagType.UNVERIFIED, 0);
        }
        
        // Count each flag type
        uint256[5] memory counts;
        for (uint i = 0; i < annotations.length; i++) {
            counts[uint(annotations[i].flagType)]++;
        }
        
        // Find most common
        uint256 maxCount = 0;
        FlagType consensus = FlagType.UNVERIFIED;
        
        for (uint i = 0; i < 5; i++) {
            if (counts[i] > maxCount) {
                maxCount = counts[i];
                consensus = FlagType(i);
            }
        }
        
        return (consensus, maxCount);
    }
}
