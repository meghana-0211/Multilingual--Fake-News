// migrations/2_deploy_contracts.js
// This tells Truffle how to deploy your smart contracts

const PublisherRegistry = artifacts.require("PublisherRegistry");
const ArticleRegistry = artifacts.require("ArticleRegistry");
const AnnotationRegistry = artifacts.require("AnnotationRegistry");

module.exports = async function(deployer) {
  console.log("\n🚀 Starting deployment...\n");
  
  // Step 1: Deploy PublisherRegistry
  console.log("1️⃣  Deploying PublisherRegistry...");
  await deployer.deploy(PublisherRegistry);
  const publisherRegistry = await PublisherRegistry.deployed();
  console.log("   ✓ PublisherRegistry deployed at:", publisherRegistry.address);
  
  // Step 2: Deploy ArticleRegistry (needs PublisherRegistry address)
  console.log("\n2️⃣  Deploying ArticleRegistry...");
  await deployer.deploy(ArticleRegistry, publisherRegistry.address);
  const articleRegistry = await ArticleRegistry.deployed();
  console.log("   ✓ ArticleRegistry deployed at:", articleRegistry.address);
  
  // Step 3: Deploy AnnotationRegistry
  console.log("\n3️⃣  Deploying AnnotationRegistry...");
  await deployer.deploy(AnnotationRegistry);
  const annotationRegistry = await AnnotationRegistry.deployed();
  console.log("   ✓ AnnotationRegistry deployed at:", annotationRegistry.address);
  
  // Print summary
  console.log("\n" + "=".repeat(70));
  console.log("🎉 DEPLOYMENT COMPLETE!");
  console.log("=".repeat(70));
  console.log("\n📋 COPY THESE ADDRESSES (you'll need them!):\n");
  console.log("PublisherRegistry:  ", publisherRegistry.address);
  console.log("ArticleRegistry:    ", articleRegistry.address);
  console.log("AnnotationRegistry: ", annotationRegistry.address);
  console.log("\n" + "=".repeat(70));
  console.log("\n💾 Save these addresses in a text file!\n");
};