# 🛰️ Bloomy ETL

**Efficient Satellite Image Processing for Remote Sensing**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NASA Space Apps 2025](https://img.shields.io/badge/NASA-Space%20Apps%202025-red.svg)](https://www.spaceappschallenge.org/)

---

## 🌍 Overview

**Bloomy ETL** is a lightweight Python tool developed for **NASA Space Apps Challenge 2025** that automates satellite image processing for vegetation monitoring. It downloads and processes **NASA's HLS (Harmonized Landsat Sentinel-2)** data to calculate **NDVI** and **EVI** vegetation indices.

**Why Bloomy?**
- ✅ **Fast**: Asynchronous downloads (10x faster than manual)
- ✅ **Simple**: One command to process months of data
- ✅ **Free**: 100% open-source
- ✅ **Smart**: Automatic cloud masking and quality filtering

**Use Cases**: Forest monitoring, fire detection, agriculture, drought analysis, urban planning.

---

## 🚀 Quick Start

### **1. Installation**

```bash
# Clone repository
git clone https://github.com/TellusOne/bloomy-etl
cd bloomy-etl

# Install dependencies
pip install -e .

# Configure NASA EarthData credentials
cp .env.example .env
# Edit .env with your username/password from https://urs.earthdata.nasa.gov
```

### **2. Create Area of Interest (AOI)**

Use [geojson.io](https://geojson.io/) to draw your study area and save as `data/my_area.geojson`

### **3. Run Pipeline**

```bash
python main.py \
  --aoi data/my_area.geojson \
  --start 2024-01-01 \
  --end 2024-06-30 \
  --output data/output.nc \
  --cloud-cover 30
```

### **4. Visualize Results**

```bash
# Interactive viewer
python tools/visualize_dataset.py data/output.nc --mode interactive

# Export to ArcGIS/QGIS
python main.py --output data/output.nc --only-export --downsample 2
```

---

## 📚 Key Features

### **Core Capabilities**
- **Automated HLS Data Download**: Query NASA EarthData for Landsat 8/9 and Sentinel-2
- **Cloud Masking**: Automatic quality filtering using Fmask
- **Vegetation Indices**: Calculate NDVI and EVI
- **Event Detection**: Identify fires, floods, droughts
- **GeoTIFF Export**: Ready for ArcGIS Pro/QGIS
- **Interactive Visualization**: Built-in time series viewer

### **Technical Features**
- Asynchronous processing (10 simultaneous downloads)
- NetCDF output (CF-compliant)
- Fixed WGS84 grid (no dimension mismatches)
- Multi-stage quality control
- Smart caching (resume interrupted downloads)

---

## 🛠️ Usage Examples

### **Forest and Phenology Monitoring**
```bash
python main.py \
  --aoi data/amazon_forest.geojson \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --output data/forest_2024.nc \
  --cloud-cover 40 \
  --export-geotiff
```

### **Agriculture Monitoring**
```bash
python main.py \
  --aoi data/farm_field.geojson \
  --start 2024-03-01 \
  --end 2024-09-30 \
  --output data/crop_season.nc \
  --cloud-cover 20
```

### **Fire Detection**
```bash
python main.py \
  --aoi data/fire_zone.geojson \
  --start 2024-07-01 \
  --end 2024-07-31 \
  --output data/fire_july.nc \
  --cloud-cover 50
```

---

## 📁 Project Structure

```
bloomy/etl/
├── main.py                      # Main entry point
├── .env.example                 # Environment template
├── requirements.txt             # Dependencies
│
├── core/                        # Core modules
│   ├── pipeline.py              # Main orchestrator
│   ├── processor.py             # Granule processing
│   ├── searcher.py              # CMR API queries
│   ├── merger.py                # Dataset merging
│   ├── quality.py               # Quality filters + event detection
│   └── authenticator.py         # NASA authentication
│
├── tools/                       # Utilities
│   ├── arcgis.py                # GeoTIFF exporter
│   └── visualize_dataset.py     # Interactive viewer
│
└── config/
    └── settings.py              # Quality thresholds
```

---

## ⚙️ Command Reference

### **Main Pipeline**

```bash
python main.py \
  --aoi <geojson>          # Area of Interest (required)
  --start <YYYY-MM-DD>     # Start date (required)
  --end <YYYY-MM-DD>       # End date (required)
  --output <file.nc>       # Output NetCDF (required)
  --cloud-cover <0-100>    # Max cloud % (default: 20)
  --batch-size <int>       # Parallel downloads (default: 10)
  --export-geotiff         # Export to GeoTIFF
  --downsample <1|2|4>     # Resolution (1=30m, 2=60m, 4=120m)
```

### **Visualization**

```bash
python tools/visualize_dataset.py <file.nc> [OPTIONS]

# Modes:
--mode interactive           # Slider navigation (recommended)
--mode single --time-idx 5   # Single timestamp
--mode timeseries --pixel-y 50 --pixel-x 100  # Pixel time series
--mode mean                  # Temporal average
--mode std                   # Temporal variability
--mode export                # Export all frames as PNG
--mode gif                   # Create animated GIF
```

### **Export to ArcGIS**

```bash
# Option 1: From main pipeline
python main.py --output data/results.nc --only-export --downsample 2

# Option 2: Standalone exporter
python tools/arcgis.py --input data/results.nc --output arcgis_export/
```

---

## 🐛 Troubleshooting

### **Authentication Failed**
- Verify credentials at [https://urs.earthdata.nasa.gov/](https://urs.earthdata.nasa.gov/)
- Check `.env` file (no typos, no quotes)
- Accept EULA at [https://search.earthdata.nasa.gov/](https://search.earthdata.nasa.gov/) (search "HLS")

### **No Granules Found**
- Expand AOI (minimum 10km × 10km)
- Increase `--cloud-cover 50`
- Check data availability at [NASA Earthdata Search](https://search.earthdata.nasa.gov/)

### **Memory Error**
- Reduce `--batch-size 5`
- Use `--downsample 2` (60m resolution)
- Process 3-6 months at a time

### **Slow Downloads**
- Enable caching: `--cache-dir cache/ --keep-cache`
- Use faster internet connection
- Process during off-peak hours

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

**NASA Space Apps Challenge 2025** - This project was developed as part of the NASA Space Apps Challenge to democratize satellite image processing.

- **NASA EarthData** - Free HLS data
- **USGS** - Landsat program
- **ESA** - Sentinel-2 program
- **Open-source community** - xarray, rasterio, matplotlib

---


---

**Made with ❤️ for NASA Space Apps Challenge 2025**

*Bloomy ETL - Making satellite image processing accessible to everyone.*
