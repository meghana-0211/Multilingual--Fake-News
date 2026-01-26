// blockchain/contracts/AnnotationRegistry.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AnnotationRegistry {
    enum FlagType { MISLEADING, FALSE, SATIRE, UNVERIFIED }
    
    struct Annotation {
        bytes32 articleHash;
        address factChecker;
        FlagType flagType;
        string ipfsHash;  // Link to detailed correction on IPFS
        uint256 timestamp;
        uint256 confidence;  // 0-100
    }
    
    mapping(bytes32 => Annotation[]) public articleAnnotations;
    mapping(address => bool) public verifiedFactCheckers;
    
    address public admin;
    
    event AnnotationAdded(bytes32 indexed articleHash, address indexed factChecker, FlagType flagType);
    event FactCheckerVerified(address indexed factChecker);
    
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin");
        _;
    }
    
    modifier onlyVerifiedFactChecker() {
        require(verifiedFactCheckers[msg.sender], "Not verified fact-checker");
        _;
    }
    
    constructor() {
        admin = msg.sender;
    }
    
    function verifyFactChecker(address _factChecker) public onlyAdmin {
        verifiedFactCheckers[_factChecker] = true;
        emit FactCheckerVerified(_factChecker);
    }
    
    function addAnnotation(
        bytes32 _articleHash,
        FlagType _flagType,
        string memory _ipfsHash,
        uint256 _confidence
    ) public onlyVerifiedFactChecker {
        require(_confidence <= 100, "Confidence must be 0-100");
        
        Annotation memory newAnnotation = Annotation({
            articleHash: _articleHash,
            factChecker: msg.sender,
            flagType: _flagType,
            ipfsHash: _ipfsHash,
            timestamp: block.timestamp,
            confidence: _confidence
        });
        
        articleAnnotations[_articleHash].push(newAnnotation);
        emit AnnotationAdded(_articleHash, msg.sender, _flagType);
    }
    
    function getAnnotations(bytes32 _articleHash) public view returns (Annotation[] memory) {
        return articleAnnotations[_articleHash];
    }
    
    function getAnnotationCount(bytes32 _articleHash) public view returns (uint256) {
        return articleAnnotations[_articleHash].length;
    }
}