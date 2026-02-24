# 👁️ Trinetra V5.0 - Multimodal Neural Registry

## Overview
Trinetra V5.0 is an advanced AI-powered multimodal asset registry system built for Bharat's digital infrastructure. It uses state-of-the-art neural embeddings (CLIP for images, CLAP for audio) with FAISS indexing for lightning-fast similarity search.

## 🆕 What's New in V5.0

### 1. **Authentication System**
- Secure login with password hashing (SHA-256)
- Role-based access control (Admin, Uploader, Viewer)
- Default credentials: `admin` / `admin123`
- Guest access option for quick demos
- User management dashboard for admins

### 2. **Quality Analysis**
- Automatic quality scoring for uploaded assets
- Image quality metrics (resolution, sharpness)
- Audio quality metrics (RMS energy, sample rate)
- Quality indicators displayed with search results

### 3. **Smart Search Suggestions**
- Real-time search suggestions based on history
- Popular searches display
- Query autocomplete functionality
- Context-aware recommendations

### 4. **Asset Clustering**
- K-means clustering visualization
- Group similar assets automatically
- Interactive cluster exploration
- Configurable cluster count (2-10)

### 5. **Enhanced Duplicate Detection**
- Smart duplicate checking during upload
- Similarity threshold configuration
- Visual comparison of duplicates
- Option to proceed with similar uploads

### 6. **Hybrid Search with Filters**
- Advanced filtering (language, tags, date, quality)
- Reranking by quality + relevance
- Minimum score thresholds
- Multi-criteria search

### 7. **Collaborative Features**
- Comments on assets
- Star ratings (1-5 stars)
- User feedback tracking
- Average rating display

### 8. **Improved UX**
- Live file preview before upload
- File size and type information
- Quality metrics in results
- Side-by-side comparison tool
- Search history with re-run capability

### 9. **Export Options**
- ZIP export (full registry backup)
- JSON export (portable format)
- Asset metadata included
- Easy migration support

### 10. **Admin Dashboard**
- User management
- System statistics
- Popular searches analytics
- User activity tracking

### 11. **Enhanced Analytics**
- Track searches by user
- 7-day usage statistics
- Average results and speed metrics
- Query frequency analysis

### 12. **Asset Metadata**
- Uploader tracking
- Quality scores
- Upload timestamps
- Collection grouping
- Custom tags and descriptions

## 🚀 Installation

### Requirements
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-org/trinetra.git
cd trinetra
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run TrinetraV5.0.py
```

4. **Access the app**
Open your browser and navigate to `http://localhost:8501`

## 📋 User Roles

### Admin
- Full system access
- User management
- Export/import data
- View analytics
- Upload assets

### Uploader
- Upload assets
- View registry
- Search functionality
- No user management

### Viewer
- Search functionality
- View assets
- Rate and comment
- No upload capability

### Guest
- Read-only access
- Basic search
- No upload or admin features

## 🎯 Key Features

### Visual Search
1. **Text Query**: Describe images in any language
2. **Image Match**: Upload an image to find similar ones
3. **Advanced Filters**: Language, tags, quality, date range

### Acoustic Search
1. **Description**: Describe sounds in natural language
2. **Audio Sample**: Upload audio for similarity search
3. **Multi-language**: Cross-lingual search support

### Neural Auditor
- Embedding visualization (2D/3D)
- PCA-based dimensionality reduction
- Asset manifest view
- Cache statistics

### Asset Management
- Batch upload support
- Automatic duplicate detection
- Quality analysis
- Collection organization
- Tag-based categorization

## 🔐 Security

- Password hashing with SHA-256
- Session-based authentication
- Role-based access control
- Secure file handling
- Input sanitization

## 📊 Architecture

### Technology Stack
- **Frontend**: Streamlit
- **ML Models**: CLIP (OpenAI), CLAP (LAION)
- **Vector DB**: FAISS (Facebook AI)
- **Database**: SQLite
- **Audio Processing**: Librosa
- **Image Processing**: Pillow

### Data Flow
1. User uploads asset
2. Quality analysis performed
3. Neural embedding generated
4. Duplicate check executed
5. FAISS index updated
6. Metadata stored in SQLite

### File Structure
```
trinetra_registry/
├── storage/          # Uploaded assets
├── image/            # Image FAISS index
├── audio/            # Audio FAISS index
├── metadata.db       # Asset metadata
├── analytics.db      # Usage analytics
├── users.db          # User accounts
└── logs/             # Application logs
```

## 🎨 Theming

Trinetra V5.0 features a beautiful dual-theme system:

### Dark Mode (Default)
- Deep blue-black background
- Golden accent colors
- High contrast for readability

### Light Mode
- Clean white panels
- Warm brown accents
- Optimal for bright environments

Toggle between themes using the sidebar.

## 📈 Performance

- **Search Speed**: ~45-100ms average
- **Embedding Cache**: 1000 vectors cached
- **Batch Upload**: Process multiple files efficiently
- **GPU Support**: Automatic CUDA detection

## 🌍 Multilingual Support

Supported languages:
- English (en)
- Hindi (hi)
- Tamil (ta)
- Telugu (te)
- Kannada (kn)
- Malayalam (ml)
- Bengali (bn)
- Marathi (mr)
- Gujarati (gu)
- Punjabi (pa)

Queries are automatically translated to English for embedding generation.

## 📝 Usage Examples

### Upload Single Asset
1. Log in as admin/uploader
2. Select modality (image/audio)
3. Choose language
4. Add tags (optional)
5. Upload file
6. Preview before registration
7. Click "Register Asset"

### Batch Upload
1. Enable "Batch Upload Mode"
2. Add tags for all files
3. Set collection name
4. Upload multiple files
5. Review results
6. Duplicates are automatically detected

### Search for Assets
1. Select "Visual Search" or "Acoustic Search"
2. Choose input mode (text or file)
3. Apply filters (optional)
4. Execute search
5. View results with quality scores
6. Rate and comment on results

### Compare Results
1. After search, expand "Compare Results"
2. Select two assets
3. View side-by-side comparison
4. See scores and metadata

## 🔧 Configuration

Edit `Config` class in the code to customize:

```python
class Config:
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
    EMBEDDING_DIM = 512
    AUDIO_DURATION_S = 7.0
    CONFIDENCE_HIGH = 0.65
    CONFIDENCE_MED = 0.45
    DUPLICATE_THRESHOLD = 0.95
    CACHE_SIZE = 1000
```

## 🐛 Troubleshooting

### FAISS not found
```bash
pip install faiss-cpu --break-system-packages
```

### GPU not detected
Check CUDA installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Models not downloading
Ensure internet connection and sufficient disk space (~2GB).

### Database locked
Close other instances of the application.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 👥 Team

**Created by Team Human**

For questions or support, contact: team@trinetra.ai

## 🙏 Acknowledgments

- OpenAI for CLIP model
- LAION for CLAP model
- Facebook AI for FAISS
- Streamlit team for the framework
- Anthropic for Claude assistance

## 📚 Documentation

For detailed API documentation, see [DOCS.md](DOCS.md)

## 🗺️ Roadmap

### V6.0 (Planned)
- [ ] Video asset support
- [ ] Real-time collaboration
- [ ] Advanced analytics dashboard
- [ ] Mobile app
- [ ] API endpoints
- [ ] Asset versioning
- [ ] Automated backups
- [ ] Custom model training
- [ ] Integration with cloud storage
- [ ] Advanced security features

---

**Version**: 5.0.0  
**Last Updated**: 2024  
**Status**: Production Ready
