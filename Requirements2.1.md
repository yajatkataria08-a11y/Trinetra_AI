# 📋 Trinetra — System Requirements Document

## 1. Purpose

This document defines the **functional and non-functional requirements**
for **Trinetra**, a multimodal asset registry and verification system built
for the **AI for Bharat Hackathon**.

The goal is to clearly specify **what the system must do**, **what it must not do**,
and **the constraints under which it operates**, independent of implementation details.

---

## 2. Stakeholders

| Stakeholder | Description |
|-----------|------------|
| End Users | Government officials, archivists, field officers |
| Administrators | Registry managers and auditors |
| Developers | Contributors and maintainers |
| Reviewers | Hackathon judges and evaluators |

---

## 3. In-Scope Capabilities

- Multimodal asset registration (image, audio)
- Cross-lingual text-based search
- Similarity-based retrieval
- Persistent asset registry
- Interactive user interface

---

## 4. Out-of-Scope Capabilities

- Biometric identification or authentication
- Real-time streaming analysis
- Model training or fine-tuning
- Automated decision-making or enforcement

---

## 5. Functional Requirements

### FR-1 Asset Registration
The system shall allow users to register image and audio assets into
a persistent registry using a unique asset identifier.

### FR-2 File Validation
The system shall validate uploaded files for:
- Maximum file size
- Allowed file formats
- Non-empty and readable content

### FR-3 Asset Identification
The system shall prevent duplicate asset identifiers within the same modality.

### FR-4 Multimodal Embedding
The system shall generate fixed-length vector embeddings for:
- Images
- Audio files
- Text queries

### FR-5 Cross-Lingual Text Support
The system shall accept text queries in regional Indian languages and
translate them to a common processing language.

### FR-6 Similarity Search
The system shall perform similarity-based search using vector embeddings
and return ranked results.

### FR-7 Top-K Retrieval
The system shall allow users to specify the number of top matching results
to retrieve.

### FR-8 Result Presentation
The system shall display:
- Asset identifiers
- Similarity scores
- Visual or audio previews where applicable

### FR-9 Persistent Storage
The system shall persist registered assets and vector indexes across
application restarts.

### FR-10 Error Handling
The system shall provide clear error messages for invalid inputs,
processing failures, or system errors.

---

## 6. Non-Functional Requirements

### NFR-1 Performance
- The system should return search results within a few seconds
  for small to medium registries.
- GPU acceleration shall be used when available.

### NFR-2 Scalability
- The system should support incremental growth in the number of registered assets.
- The architecture shall allow migration to approximate nearest neighbor search.

### NFR-3 Reliability
- The system shall not corrupt registry data during normal operation.
- Temporary files shall be cleaned after processing.

### NFR-4 Usability
- The user interface shall be intuitive and require minimal training.
- The system shall provide visual feedback during long-running operations.

### NFR-5 Security
- User-supplied inputs shall be validated and sanitized.
- Direct access to internal file paths shall be restricted.

### NFR-6 Portability
- The system shall run on CPU-only environments.
- GPU usage shall be optional, not mandatory.

### NFR-7 Maintainability
- Code shall be modular and readable.
- System behavior shall be documented in design artifacts.

---

## 7. Constraints

- Must use open or inspectable AI models
- Must run on commodity hardware
- Must operate without proprietary cloud dependencies
- Must prioritize explainability over black-box automation

---

## 8. Assumptions

- Internet access may be intermittent in deployment environments
- Text translation services may occasionally fail
- Asset registries will initially be moderate in size

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|----|-----------|
| Short audio clips produce weak embeddings | Minimum duration enforced |
| Large registries slow down search | FAISS index upgrade path |
| Translation failures | Fallback to original text |
| Misinterpretation of similarity scores | Clear UI guidance |

---

## 10. Compliance and Ethics

- The system does not perform biometric identification
- Outputs are advisory, not authoritative
- Designed for transparency and auditability
- Intended for public-good and governance use cases

---

## 11. Success Criteria

The system will be considered successful if it:
- Allows reliable multimodal search
- Supports regional language queries
- Persists assets and indexes correctly
- Demonstrates clear relevance to Bharat-scale use cases

---

## 12. Future Requirements (Not Implemented)

- Role-based access control
- Audit trails and usage logs
- API-based ingestion
- Integration with digital signature frameworks
