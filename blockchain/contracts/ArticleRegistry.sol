// blockchain/contracts/ArticleRegistry.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./PublisherRegistry.sol";

contract ArticleRegistry {
    struct Article {
        bytes32 contentHash;
        address publisher;
        uint256 timestamp;
        string language;
        bool exists;
    }
    
    mapping(bytes32 => Article) public articles;
    bytes32[] public articleHashes;
    
    PublisherRegistry public publisherRegistry;
    
    event ArticleRegistered(bytes32 indexed contentHash, address indexed publisher, string language);
    
    constructor(address _publisherRegistry) {
        publisherRegistry = PublisherRegistry(_publisherRegistry);
    }
    
    function registerArticle(bytes32 _contentHash, string memory _language) public {
        require(publisherRegistry.isRegistered(msg.sender), "Publisher not registered");
        require(!articles[_contentHash].exists, "Article already registered");
        
        articles[_contentHash] = Article({
            contentHash: _contentHash,
            publisher: msg.sender,
            timestamp: block.timestamp,
            language: _language,
            exists: true
        });
        
        articleHashes.push(_contentHash);
        emit ArticleRegistered(_contentHash, msg.sender, _language);
    }
    
    function verifyArticle(bytes32 _contentHash) public view returns (bool, address, uint256) {
        Article memory article = articles[_contentHash];
        return (article.exists, article.publisher, article.timestamp);
    }
    
    function getArticle(bytes32 _contentHash) public view returns (Article memory) {
        return articles[_contentHash];
    }
}