import cartopy.io.shapereader as shp
shp_path = shp.natural_earth(
    resolution='10m',
    category='physical',
    name='coastline',
    data_dir='data/shapefiles'
)
print(f"Shapefile downloaded to: {shp_path}")
