#!/usr/bin/env python3
"""
Test script for fingerprint-based SAR prediction in admet.py

Demonstrates the upgrade from MW/logP guesswork to structural similarity.
"""

from bodysim_engine.engine.admet import (
    build_drug_profile, HAS_RDKIT, GOLD_STANDARD_FINGERPRINTS
)

print("=" * 80)
print("FINGERPRINT-BASED SAR PREDICTION TEST")
print("=" * 80)
print(f"\nRDKit Available: {HAS_RDKIT}")
if HAS_RDKIT:
    print(f"Gold Standard Fingerprints Loaded: {len(GOLD_STANDARD_FINGERPRINTS)} transporters")
    for trans, substrates in GOLD_STANDARD_FINGERPRINTS.items():
        print(f"  - {trans}: {len(substrates)} reference substrates")
else:
    print("⚠️  RDKit not installed. Will use Gaussian SAR fallback.")
    print("   To enable fingerprint matching: pip install rdkit")

print("\n" + "=" * 80)
print("TEST 1: WITHOUT SMILES (uses Gaussian SAR fallback)")
print("=" * 80)

profile_no_smiles = build_drug_profile(
    name="Test Drug (No SMILES)",
    logp=2.5,
    fup=0.1,
    mw=350,
    pka=4.5,
    drug_type="acidic"
)

print(f"\nDrug: {profile_no_smiles['name']}")
print(f"Transporter prediction method: gaussian (no SMILES provided)")
print(f"\nRenal Transport Substrates:")
for trans_name, trans_info in profile_no_smiles["renal_transport"].items():
    print(f"  {trans_name}:")
    print(f"    - Probability: {trans_info['probability']:.3f}")
    print(f"    - Method: {trans_info.get('method', 'N/A')}")
    print(f"    - Similarity: {trans_info.get('similarity', 'N/A')}")

if HAS_RDKIT:
    print("\n" + "=" * 80)
    print("TEST 2: WITH SMILES (uses fingerprint matching!)")
    print("=" * 80)
    
    # Metformin SMILES - should be HIGH similarity to OCT2 gold standard
    metformin_smiles = "CN(C)C(=N)NC"
    
    profile_with_smiles = build_drug_profile(
        name="Metformin-like Drug",
        logp=-1.43,
        fup=0.97,
        mw=129.16,
        pka=11.5,
        drug_type="basic",
        smiles=metformin_smiles
    )
    
    print(f"\nDrug: {profile_with_smiles['name']}")
    print(f"SMILES: {metformin_smiles}")
    print(f"Note: This is the actual Metformin SMILES - should see HIGH fingerprint similarity!\n")
    
    print(f"Renal Transport Substrates:")
    for trans_name, trans_info in profile_with_smiles["renal_transport"].items():
        method = trans_info.get('method', 'N/A')
        similarity = trans_info.get('similarity')
        prob = trans_info['probability']
        
        print(f"  {trans_name}:")
        print(f"    - Probability: {prob:.3f}")
        print(f"    - Method: {method}")
        if similarity is not None:
            print(f"    - Tanimoto Similarity: {similarity:.3f} ★ (Fingerprint-based!)")
        else:
            print(f"    - Similarity: N/A (fallback to Gaussian)")
    
    print("\n" + "─" * 80)
    print("TEST 3: STRUCTURALLY DISSIMILAR MOLECULE")
    print("─" * 80)
    
    # A highly dissimilar SMILES - should show LOW fingerprint similarity
    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    
    profile_aspirin = build_drug_profile(
        name="Aspirin (Dissimilar to Metformin)",
        logp=1.19,
        fup=0.99,
        mw=180.16,
        pka=3.49,
        drug_type="acidic",
        smiles=aspirin_smiles
    )
    
    print(f"\nDrug: {profile_aspirin['name']}")
    print(f"SMILES: {aspirin_smiles}")
    print(f"Note: Aspirin is structurally different from Metformin - should see LOW fingerprint similarity\n")
    
    print(f"Renal Transport Substrates:")
    for trans_name, trans_info in profile_aspirin["renal_transport"].items():
        method = trans_info.get('method', 'N/A')
        similarity = trans_info.get('similarity')
        prob = trans_info['probability']
        
        print(f"  {trans_name}:")
        print(f"    - Probability: {prob:.3f}")
        print(f"    - Method: {method}")
        if similarity is not None:
            print(f"    - Tanimoto Similarity: {similarity:.3f}")
        else:
            print(f"    - Similarity: N/A (fallback to Gaussian)")

print("\n" + "=" * 80)
print("✓ Test completed!")
print("=" * 80)
