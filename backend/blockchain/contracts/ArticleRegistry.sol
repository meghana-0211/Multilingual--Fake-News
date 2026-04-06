// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * ARTICLE REGISTRY CONTRACT
 * 
 * Think: Library catalog - records when articles were published
 * 
 * What it stores:
 * - Article hash (unique fingerprint of content)
 * - Publisher who posted it
 * - Timestamp when posted
 * - Language (hindi/gujarati/marathi/telugu)
 * 
 * Why hash instead of full text?
 * - Blockchain storage is expensive
 * - Hash = 32 bytes vs Article = thousands of bytes
 * - Hash uniquely identifies article
 * - Can verify "this article was posted on this date"
 */

import "./PublisherRegistry.sol";

contract ArticleRegistry {
    
    // ============================================
    // DATA STRUCTURES
    // ============================================
    
    /**
     * Article struct - Metadata about an article
     */
    struct Article {
        bytes32 contentHash;      // SHA-256 hash of article text
        address publisher;        // Who published it
        uint256 timestamp;        // When published
        string language;          // hindi/gujarati/marathi/telugu
        bool exists;              // Does this article exist?
    }
    
    // Mapping: contentHash → Article
    // Like: {'0xa3f2...': Article(...)}
    mapping(bytes32 => Article) public articles;
    
    // Array of all article hashes
    bytes32[] public articleHashes;
    
    // Reference to PublisherRegistry (to verify publishers)
    PublisherRegistry public publisherRegistry;
    
    // ============================================
    // EVENTS
    // ============================================
    
    event ArticleRegistered(
        bytes32 indexed contentHash,
        address indexed publisher,
        string language,
        uint256 timestamp
    );
    
    // ============================================
    // CONSTRUCTOR
    // ============================================
    
    /**
     * @param _publisherRegistry Address of deployed PublisherRegistry
     */
    constructor(address _publisherRegistry) {
        publisherRegistry = PublisherRegistry(_publisherRegistry);
    }
    
    // ============================================
    // FUNCTIONS
    // ============================================
    
    /**
     * Register a new article
     * 
     * Flow:
     * 1. Publisher calls this with article hash
     * 2. Contract checks: Is caller a verified publisher?
     * 3. Contract checks: Was this article already registered?
     * 4. Contract stores: hash, publisher, time, language
     * 
     * @param _contentHash SHA-256 hash of article (32 bytes)
     * @param _language Language code (e.g., "hindi")
     * 
     * Example from Python:
     * hash = hashlib.sha256(article_text.encode()).digest()
     * contract.registerArticle(hash, "hindi")
     */
    function registerArticle(bytes32 _contentHash, string memory _language) 
        public 
    {
        // Only verified publishers can register articles
        require(
            publisherRegistry.isVerified(msg.sender),
            "You must be a verified publisher"
        );
        
        // Article can't already exist
        require(
            !articles[_contentHash].exists,
            "Article already registered"
        );
        
        // Create article record
        articles[_contentHash] = Article({
            contentHash: _contentHash,
            publisher: msg.sender,
            timestamp: block.timestamp,  // Current blockchain time
            language: _language,
            exists: true
        });
        
        // Add to array
        articleHashes.push(_contentHash);
        
        // Emit event
        emit ArticleRegistered(
            _contentHash,
            msg.sender,
            _language,
            block.timestamp
        );
    }
    
    /**
     * Verify if article exists and get details
     * 
     * @param _contentHash Article hash to check
     * @return exists Does it exist?
     * @return publisher Who published it?
     * @return timestamp When was it published?
     * 
     * Anyone can call this (public verification)
     * 
     * Example:
     * hash = hashlib.sha256("fake article text".encode()).digest()
     * exists, publisher, time = contract.verifyArticle(hash)
     * if exists:
     *     print(f"Article was published by {publisher} on {time}")
     */
    function verifyArticle(bytes32 _contentHash) 
        public 
        view 
        returns (
            bool exists,
            address publisher,
            uint256 timestamp
        ) 
    {
        Article memory article = articles[_contentHash];
        return (
            article.exists,
            article.publisher,
            article.timestamp
        );
    }
    
    /**
     * Get complete article details
     * 
     * @param _contentHash Article hash
     * @return Article Complete article struct
     */
    function getArticle(bytes32 _contentHash) 
        public 
        view 
        returns (Article memory) 
    {
        require(articles[_contentHash].exists, "Article not found");
        return articles[_contentHash];
    }
    
    /**
     * Get total number of registered articles
     */
    function getArticleCount() 
        public 
        view 
        returns (uint256) 
    {
        return articleHashes.length;
    }
    
    /**
     * Check if article was published by specific publisher
     * 
     * @param _contentHash Article hash
     * @param _publisher Publisher address
     * @return bool True if this publisher posted this article
     */
    function wasPublishedBy(bytes32 _contentHash, address _publisher) 
        public 
        view 
        returns (bool) 
    {
        return articles[_contentHash].publisher == _publisher;
    }
}
