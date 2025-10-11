import cartopy.io.shapereader as shp
import os

# Zielordner für die Shapefiles
shapefile_dir = "data/shapefiles"
os.makedirs(shapefile_dir, exist_ok=True)

# Download
shp_path = shp.natural_earth(
    resolution='10m',
    category='physical',
    name='coastline'
)
# Kopiere die Datei in dein Repo-Verzeichnis
import shutil
shutil.copy(shp_path, shapefile_dir)

print(f"Shapefile downloaded to: {shapefile_dir}")
