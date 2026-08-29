import sys
import cfgrib
import pandas as pd
import os
import struct
import zlib
from zoneinfo import ZoneInfo
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import gc
import matplotlib
matplotlib.use("Agg")  # headless, kein Display nötig
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import matplotlib.colors as mcolors
from PIL import Image
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------
# Eingabe-/Ausgabe
# ------------------------------
data_dir = sys.argv[1]        # z.B. "output"
output_dir = sys.argv[2]      # z.B. "output/maps"
var_type = sys.argv[3]        # 't2m', 'ww', 'tp', 'tp_acc', ...
os.makedirs(output_dir, exist_ok=True)

ignore_codes = {4}

# ------------------------------
# WW-Farben
# ------------------------------
ww_colors_base = {
    0: "#FFFFFF", 1: "#D3D3D3", 2: "#A9A9A9", 3: "#696969",
    45: "#FFFF00", 48: "#FFD700",
    56: "#FFA500", 57: "#C06A00",
    51: "#00FF00", 53: "#00C300", 55: "#009700",
    61: "#00FF00", 63: "#00C300", 65: "#009700",
    80: "#00FF00", 81: "#00C300", 82: "#009700",
    66: "#FF6347", 67: "#8B0000",
    71: "#ADD8E6", 73: "#6495ED", 75: "#00008B",
    85: "#ADD8E6", 86: "#6495ED",
    77: "#ADD8E6",
    95: "#FF77FF", 96: "#C71585", 99: "#C71585"
}

# ------------------------------
# Temperatur-Farben
# ------------------------------
t2m_bounds = list(range(-36, 50, 2))
t2m_colors = LinearSegmentedColormap.from_list(
    "t2m_smoooth",
    [
        "#F675F4", "#F428E9", "#B117B5", "#950CA2", "#640180",
        "#3E007F", "#00337E", "#005295", "#1292FF", "#49ACFF",
        "#8FCDFF", "#B4DBFF", "#B9ECDD", "#88D4AD", "#07A125",
        "#3FC107", "#9DE004", "#E7F700", "#F3CD0A", "#EE5505",
        "#C81904", "#AF0E14", "#620001", "#C87879", "#FACACA",
        "#E1E1E1", "#6D6D6D"
    ],
    N=len(t2m_bounds)
)
t2m_norm = BoundaryNorm(t2m_bounds, ncolors=len(t2m_bounds))

# ------------------------------
# Niederschlags-Farben 1h (tp)
# ------------------------------
prec_bounds = [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               12, 14, 16, 20, 24, 30, 40, 50, 60, 80, 100, 125]
prec_colors = ListedColormap([
    "#B4D7FF", "#75BAFF", "#349AFF", "#0582FF", "#0069D2",
    "#003680", "#148F1B", "#1ACF06", "#64ED07", "#FFF32B",
    "#E9DC01", "#F06000", "#FF7F26", "#FFA66A", "#F94E78",
    "#F71E53", "#BE0000", "#880000", "#64007F", "#C201FC",
    "#DD66FE", "#EBA6FF", "#F9E7FF", "#D4D4D4", "#969696"
])
prec_colors.set_under(alpha=0)
prec_norm = mcolors.BoundaryNorm(prec_bounds, prec_colors.N)

# ------------------------------
# Aufsummierter Niederschlag (tp_acc)
# ------------------------------
tp_acc_bounds = [0.0, 0.1, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
                  125, 150, 175, 200, 250, 300, 400, 500]
tp_acc_colors = ListedColormap([
    "#FFFFFF", "#B4D7FF", "#75BAFF", "#349AFF", "#0582FF", "#0069D2",
    "#003680", "#148F1B", "#1ACF06", "#64ED07", "#FFF32B",
    "#E9DC01", "#F06000", "#FF7F26", "#FFA66A", "#F94E78",
    "#F71E53", "#BE0000", "#880000", "#64007F", "#C201FC",
    "#DD66FE", "#EBA6FF", "#F9E7FF", "#D4D4D4", "#969696"
])
tp_acc_norm = mcolors.BoundaryNorm(tp_acc_bounds, tp_acc_colors.N)

# ------------------------------
# Windböen-Farben
# ------------------------------
wind_bounds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 220, 240, 260, 280, 300]
wind_colors = ListedColormap([
    "#68AD05", "#8DC00B", "#B1D415", "#D5E81C", "#FBFC22",
    "#FAD024", "#F9A427", "#FC7929", "#FB4D2B", "#EA2B57",
    "#FB22A5", "#FC22CE", "#FC22F5", "#FC62F8", "#FD80F8",
    "#FFBFFC", "#FEDFFE", "#FEFFFF", "#E1E0FF", "#C3C3FF",
    "#A5A5FF", "#A5A5FF", "#6868FE"
])
wind_norm = mcolors.BoundaryNorm(wind_bounds, wind_colors.N)

# ------------------------------
# Schneehöhen-Farben
# ------------------------------
snow_bounds = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200, 250, 300, 400]
snow_colors = ListedColormap([
    "#F8F8F8", "#DCDBFA", "#AAA9C8", "#75BAFF", "#349AFF", "#0582FF",
    "#0069D2", "#004F9C", "#01327F", "#4B007F", "#64007F", "#9101BB",
    "#C300FC", "#D235FF", "#EBA6FF", "#F4CEFF", "#FAB2CA", "#FF9798",
    "#FE6E6E", "#DF093F", "#BE0000", "#A40000", "#880000", "#460000"
])
snow_norm = mcolors.BoundaryNorm(snow_bounds, snow_colors.N)

# ------------------------------
# Schneefallgrenze (SNOWLMT)
# ------------------------------
snowfall_bounds = [0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000]
snowfall_colors = ListedColormap([
    "#FF00A6", "#D900FF", "#8C00FF", "#0008FF", "#0099FF",
    "#00F2FF", "#1AFF00", "#FFFB00", "#FFBF00", "#FFA600",
    "#FF6F00", "#930000",
])
snowfall_norm = mcolors.BoundaryNorm(snowfall_bounds, snowfall_colors.N)

# ------------------------------
# Luftdruck
# ------------------------------
pmsl_bounds_colors = list(range(912, 1070, 4))
pmsl_colors = LinearSegmentedColormap.from_list(
    "pmsl_smooth",
    [
        "#FF6DFF", "#C418C4", "#950CA2", "#5A007D", "#3D007F",
        "#00337E", "#0472CB", "#4FABF8", "#A3D4FF", "#79DAAD",
        "#07A220", "#3EC008", "#9EE002", "#F3FC01", "#F19806",
        "#F74F11", "#B81212", "#8C3234", "#C87879", "#F9CBCD",
        "#E2E2E2"
    ],
    N=len(pmsl_bounds_colors)
)
pmsl_norm = BoundaryNorm(pmsl_bounds_colors, ncolors=len(pmsl_bounds_colors))

# ------------------------------
# DBZ-CMAX Farben
# ------------------------------
dbz_bounds = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 63, 67, 70]
dbz_colors = ListedColormap([
    "#FFFFFF", "#B3EFED", "#8CE7E2", "#00F5ED",
    "#00CEF0", "#01AFF4", "#028DF6", "#014FF7", "#0000F6",
    "#00FF01", "#01DF00", "#00D000", "#00BF00", "#00A701",
    "#019700", "#FFFF00", "#F9F000", "#EDD200", "#E7B500",
    "#FF5000", "#FF2801", "#F40000", "#EA0001", "#CC0000",
    "#FFC8FF", "#E9A1EA", "#D379D3", "#BE55BE", "#960E96"
])
dbz_colors.set_under(alpha=0)
dbz_norm = mcolors.BoundaryNorm(dbz_bounds, dbz_colors.N)

# ------------------------------
# Gesamtwassergehalt (TWATER)
# ------------------------------
twater_bounds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
twater_colors = ListedColormap([
    "#6E4A00", "#B49E62", "#D7CD13", "#B9F019", "#1ACF06",
    "#08534C", "#035DBE", "#2692FF", "#75BAFF", "#CBBFFF",
    "#EBA6FF", "#DD66FE", "#AC01DD", "#7C009E", "#673775",
    "#6B6B6B", "#818181", "#969696"
])
twater_norm = mcolors.BoundaryNorm(twater_bounds, twater_colors.N)

# ------------------------------
# Kartendomäne
# ------------------------------
extent = [-3.94, 20.34, 43.18, 58.08] 

FOOTER_TEXTS = {
    "ww": "Signifikantes Wetter",
    "t2m": "Temperatur 2m (°C)",
    "t850": "Temperatur 850hPa (°C)",
    "tp": "Niederschlag, 1Std (mm)",
    "tp_acc": "Akkumulierter Niederschlag (mm)",
    "dbz_cmax": "Sim. max. Radarreflektivität (dBZ)",
    "wind": "Windböen (km/h)",
    "snow": "Schneehöhe (cm)",
    "twater": "Gesamtwassergehalt (mm)",
    "snowfall": "Schneefallgrenze (m)",
    "pmsl": "Luftdruck auf Meereshöhe (hPa)",
}

# Einheit je Variable - für die Wertanzeige im Frontend
VALUE_UNITS = {
    "ww": "",
    "t2m": "°C",
    "t850": "°C",
    "tp": "mm",
    "tp_acc": "mm",
    "dbz_cmax": "dBZ",
    "wind": "km/h",
    "snow": "cm",
    "twater": "mm",
    "snowfall": "m",
    "pmsl": "hPa",
}

# Nachkommastellen je Variable für die Wertanzeige
VALUE_DECIMALS = {
    "ww": 0,
    "t2m": 1,
    "t850": 1,
    "tp": 1,
    "tp_acc": 1,
    "dbz_cmax": 0,
    "wind": 0,
    "snow": 1,
    "twater": 1,
    "snowfall": 0,
    "pmsl": 0,
}

# Sentinel-Wert für "kein Datum/außerhalb" (nicht mehr für Binärdatei
# benötigt, aber zur Referenz im Manifest praktisch)
VALUE_NODATA = -9999.0

# ------------------------------
# EPSG:4326 -> EPSG:3857 (Web Mercator)
# ------------------------------
# Leaflet/OSM rendern intern in Web Mercator (EPSG:3857). Unsere GRIB-Daten
# liegen als Plattkarte (EPSG:4326, gleichmäßiges lon/lat-Raster) vor. Ein
# L.imageOverlay dehnt ein rohes EPSG:4326-Bild einfach linear in die
# angegebenen Lat/Lon-Bounds - das ist gerade auf größeren Ausschnitten
# sichtbar falsch (Nord-Süd-Stauchung/Streckung). Daher wird das Datenfeld
# hier vor dem Speichern explizit nach EPSG:3857 umprojiziert, sodass es
# 1:1 in die (weiterhin in Lat/Lon angegebenen) Overlay-Bounds passt.
EARTH_RADIUS = 6378137.0  # Meter, WGS84/Web-Mercator-Kugelradius
WEBMERCATOR_WIDTH = 1024   # Ziel-Bildbreite in Pixeln für die Reprojektion


def lonlat_to_webmercator(lon_deg, lat_deg):
    x = EARTH_RADIUS * np.radians(lon_deg)
    y = EARTH_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))
    return x, y


def webmercator_target_grid(extent, out_width=WEBMERCATOR_WIDTH):
    lon_min, lon_max, lat_min, lat_max = extent
    x_min, y_min = lonlat_to_webmercator(lon_min, lat_min)
    x_max, y_max = lonlat_to_webmercator(lon_max, lat_max)
    aspect = (y_max - y_min) / (x_max - x_min)
    out_height = max(int(round(out_width * aspect)), 1)
    x_new = np.linspace(x_min, x_max, out_width)
    y_new = np.linspace(y_min, y_max, out_height)  # aufsteigend: Süd -> Nord
    return x_new, y_new


def warp_equirect_to_webmercator(data, lon, lat, extent, method="linear",
                                  out_width=WEBMERCATOR_WIDTH):
    """data/lon/lat: reguläres EPSG:4326-Gitter, lon und lat aufsteigend
    sortiert. Gibt das Datenfeld auf einem regulären EPSG:3857-Pixelraster
    zurück (ebenfalls Süd -> Nord aufsteigend), zugeschnitten auf extent.
    Es wird direkt vom nativen Modellgitter gewarpt - kein zusätzlicher
    Zwischenschritt auf ein feineres EPSG:4326-Gitter mehr nötig."""
    x_new, y_new = webmercator_target_grid(extent, out_width=out_width)
    xx, yy = np.meshgrid(x_new, y_new)
    lon_grid = np.degrees(xx / EARTH_RADIUS)
    lat_grid = np.degrees(2 * np.arctan(np.exp(yy / EARTH_RADIUS)) - np.pi / 2)

    interp_func = RegularGridInterpolator(
        (lat, lon), data,
        method=method,
        bounds_error=False,
        fill_value=np.nan
    )
    pts = np.array([lat_grid.ravel(), lon_grid.ravel()]).T
    warped = interp_func(pts).reshape(lat_grid.shape)
    return warped


def data_to_rgba(data, cmap, norm):
    """Wandelt ein 2D-Datenarray in ein RGBA-uint8-Array um.
    NaN-Werte werden komplett transparent."""
    rgba = cmap(norm(data))
    rgba = (rgba * 255).astype(np.uint8)
    nan_mask = ~np.isfinite(data)
    rgba[nan_mask, 3] = 0
    return rgba


def composite_over(top_rgba, bottom_rgba):
    """Alpha-Compositing (Porter-Duff 'over'): top_rgba über bottom_rgba legen.
    Beide Arrays: uint8, shape (H,W,4), gleiche Ausrichtung."""
    top = top_rgba.astype(np.float32) / 255.0
    bottom = bottom_rgba.astype(np.float32) / 255.0
    a_top = top[..., 3:4]
    a_bottom = bottom[..., 3:4]
    out_alpha = a_top + a_bottom * (1 - a_top)
    out_rgb = np.where(
        out_alpha > 1e-6,
        (top[..., :3] * a_top + bottom[..., :3] * a_bottom * (1 - a_top)) / np.maximum(out_alpha, 1e-6),
        0.0
    )
    out = np.concatenate([out_rgb, out_alpha], axis=-1)
    return (out * 255).astype(np.uint8)


def render_contour_overlay_rgba(x_new, y_new, data_grid, main_levels, fine_levels):
    """Zeichnet Isolinien (z.B. Isobaren) mit Beschriftung auf einen
    transparenten Hintergrund, exakt im Pixelraster von (x_new, y_new)
    (Web-Mercator-Meter). Rückgabe: RGBA-uint8-Array, Zeile 0 = Süden
    (gleiche Konvention wie die Datenraster in diesem Skript)."""
    out_width, out_height = len(x_new), len(y_new)

    if not np.any(np.isfinite(data_grid)):
        return np.zeros((out_height, out_width, 4), dtype=np.uint8)

    d_min = np.nanmin(data_grid)
    d_max = np.nanmax(data_grid)
    main_lv = [lv for lv in main_levels if d_min <= lv <= d_max]
    fine_lv = [lv for lv in fine_levels if d_min <= lv <= d_max]

    dpi = 100
    fig = plt.figure(figsize=(out_width / dpi, out_height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x_new[0], x_new[-1])
    ax.set_ylim(y_new[0], y_new[-1])
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    xx, yy = np.meshgrid(x_new, y_new)

    if fine_lv:
        ax.contour(xx, yy, data_grid, levels=fine_lv, colors="gray", linewidths=0.5, alpha=0.4)
    if main_lv:
        ax.contour(xx, yy, data_grid, levels=main_lv, colors="white", linewidths=0.9, alpha=0.95)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)

    if buf.shape[1] != out_width or buf.shape[0] != out_height:
        buf = np.array(Image.fromarray(buf, mode="RGBA").resize((out_width, out_height), Image.LANCZOS))

    # Canvas-Zeile 0 = Norden (oben) -> spiegeln, damit Zeile 0 = Süden
    return buf[::-1]


def save_transparent_webp(data, cmap, norm, out_path, contour_rgba=None):
    rgba = data_to_rgba(data, cmap, norm)
    if contour_rgba is not None:
        rgba = composite_over(contour_rgba, rgba)
    img = Image.fromarray(rgba[::-1, :, :], mode="RGBA")

    # Verlustfrei speichern: die Colormaps arbeiten mit diskreten Stufen
    # und set_under(alpha=0) für Transparenz - eine verlustbehaftete
    # WebP-Kompression würde Farbgrenzen und den Transparenz-Threshold
    # sichtbar verwischen.
    #
    # method=4 statt 6: bei diesen Bildern (große einfarbige/transparente
    # Flächen, wenige diskrete Farbstufen) liefert method=6 praktisch
    # dieselbe Dateigröße wie method=4, braucht dabei aber massiv mehr
    # Laufzeit - der höhere Aufwand bringt hier also keinen Vorteil.
    img.save(out_path, format="WEBP", lossless=True, method=4)


_dom_x_min, _dom_y_min = lonlat_to_webmercator(extent[0], extent[2])
_dom_x_max, _dom_y_max = lonlat_to_webmercator(extent[1], extent[3])
DOMAIN_EXTENT_3857 = [float(_dom_x_min), float(_dom_y_min), float(_dom_x_max), float(_dom_y_max)]

# ------------------------------
# Bounding Box für den eingebetteten DVAL-Chunk (nur t2m/wind)
# ------------------------------
# Die volle Kartendomäne (inkl. Frankreich, Benelux, Polen, Tschechien etc.)
# ist für die eingebetteten Rohwerte unnötig groß - wir brauchen die Werte
# nur für Deutschland (+ etwas Rand für Grenzregionen beim Hovern). Das
# Farbbild selbst bleibt unverändert auf der vollen Domäne, nur das
# DVAL-Array wird auf dieses Rechteck zugeschnitten.
EMBED_DATA_VARS = {"t2m", "wind"}
GERMANY_BBOX_LONLAT = [5.5, 15.3, 47.0, 55.3]  # lon_min, lon_max, lat_min, lat_max

# Gleiches Ziel-Pixelraster wie in warp_equirect_to_webmercator (muss mit
# WEBMERCATOR_WIDTH übereinstimmen, damit die Indizes exakt passen) -
# einmalig außerhalb der Schleife berechnet, da pro Lauf identisch.
_full_x_new, _full_y_new = webmercator_target_grid(extent, out_width=WEBMERCATOR_WIDTH)

_gbx_min, _gby_min = lonlat_to_webmercator(GERMANY_BBOX_LONLAT[0], GERMANY_BBOX_LONLAT[2])
_gbx_max, _gby_max = lonlat_to_webmercator(GERMANY_BBOX_LONLAT[1], GERMANY_BBOX_LONLAT[3])

# Indizes im vollen Raster, die die Bbox gerade so umschließen (lieber
# ein Pixel zu viel als zu wenig - daher außen aufrunden statt clippen).
_col_i0 = max(0, np.searchsorted(_full_x_new, _gbx_min, side="left") - 1)
_col_i1 = min(len(_full_x_new) - 1, np.searchsorted(_full_x_new, _gbx_max, side="right"))
_row_i0 = max(0, np.searchsorted(_full_y_new, _gby_min, side="left") - 1)
_row_i1 = min(len(_full_y_new) - 1, np.searchsorted(_full_y_new, _gby_max, side="right"))

# Exakte Mercator-Extent des zugeschnittenen Rasters (= tatsächliche
# Gitterpunkte an den Rändern, nicht die rohe Bbox - damit die
# Rücktransformation im Frontend pixelgenau bleibt).
GERMANY_CROP_EXTENT_3857 = [
    float(_full_x_new[_col_i0]), float(_full_y_new[_row_i0]),
    float(_full_x_new[_col_i1]), float(_full_y_new[_row_i1]),
]


def crop_to_germany(data_south_first):
    """data_south_first: 2D-Array wie von warp_equirect_to_webmercator
    zurückgegeben (row0 = Süden, aufsteigend in Mercator-Y wie
    _full_y_new). Schneidet auf die Deutschland-Bbox zu."""
    return data_south_first[_row_i0:_row_i1 + 1, _col_i0:_col_i1 + 1]


def data_to_rgba(data, cmap, norm):
    """Wandelt ein 2D-Datenarray in ein RGBA-uint8-Array um.
    NaN-Werte werden komplett transparent."""
    rgba = cmap(norm(data))  # float RGBA in [0,1], shape (H,W,4)
    rgba = (rgba * 255).astype(np.uint8)
    nan_mask = ~np.isfinite(data)
    rgba[nan_mask, 3] = 0
    return rgba


# ------------------------------
# Eingebettete Rohdaten (DVAL-Chunk) im WebP
# ------------------------------
# WebP ist ein RIFF-Container. RIFF erlaubt beliebige zusätzliche Chunks
# mit eigenem FourCC-Tag - konforme Reader (Browser, Bildbetrachter, PIL)
# ignorieren unbekannte Chunks einfach, genau wie private PNG-Chunks.
# Wir hängen hier einen "DVAL"-Chunk mit den echten physikalischen Werten
# (nicht den Farben!) an, komprimiert mit zlib, plus die exakte
# Web-Mercator-Domäne in Metern - damit lässt sich im Frontend pixelgenau
# und ohne Rundungsfehler zurückrechnen, welcher Wert an welcher
# Lon/Lat-Position liegt. Keine separate Datei nötig.
#
# Chunk-Layout (nach den 8 Bytes FourCC + size), Version 2:
#   uint8  version   (=2)
#   uint8  dtype     (=1: int16 quantisiert, little-endian, zlib-komprimiert)
#   uint32 width
#   uint32 height
#   float64 x_min, y_min, x_max, y_max   (EPSG:3857, Meter)
#   float64 scale     (echter Wert = raw_int16 * scale; NaN-Sentinel = -32768)
#   ... zlib-komprimierte int16-Daten, row0 = Norden (Bild-Orientierung),
#       row-major, len = width*height*2 Bytes nach Dekompression
#
# Warum int16 statt float32: t2m/wind werden ohnehin nur mit 1 bzw. 0
# Nachkommastellen angezeigt - float32 speichert weit mehr Präzision, als
# je genutzt wird. Die Quantisierung auf ein festes Raster (z.B. 0.05°C)
# halbiert nicht nur die Rohgröße, sondern erzeugt durch den Wegfall des
# Interpolations-"Rauschens" auch deutlich mehr exakt gleiche
# Nachbarwerte - das macht das Feld für zlib erheblich kompressibler.
DVAL_FOURCC = b"DVAL"

# Quantisierungsschritt je Variable (feiner als die Anzeige-Nachkommastellen
# in VALUE_DECIMALS, damit keinerlei sichtbarer Genauigkeitsverlust entsteht).
QUANTUM_STEP = {
    "t2m": 0.05,   # °C, Anzeige mit 1 Dezimalstelle -> 0.05 ist mehr als genug
    "wind": 0.2,   # km/h, Anzeige mit 0 Dezimalstellen -> 0.2 ist mehr als genug
}
NAN_SENTINEL_I16 = -32768


def embed_data_chunk(webp_path, data, extent_3857, quantum, fourcc=DVAL_FOURCC):
    """Hängt ein rohes Datenfeld als privaten, int16-quantisierten RIFF-Chunk
    an ein WebP an.

    data: 2D-Array (float), row0 = Norden (also bereits wie fürs Bild
          gespiegelt).
    extent_3857: [x_min, y_min, x_max, y_max] in Web-Mercator-Metern -
                 exakt das Raster, auf dem `data` liegt.
    quantum: Rasterschritt in den Originaleinheiten (z.B. 0.05 für °C).
    """
    height, width = data.shape

    nan_mask = ~np.isfinite(data)
    data_filled = np.where(nan_mask, 0.0, data)  # verhindert NaN->int Warnung beim Runden/Casten
    quant = np.round(data_filled / quantum)
    # Sicherheitsclip: verhindert einen int16-Überlauf bei extremen
    # Ausreißern, ohne das eigentlich zulässige Wertespektrum
    # (t2m/wind liegen weit darunter) einzuschränken.
    quant = np.clip(quant, -32767, 32767).astype(np.int16)
    quant[nan_mask] = NAN_SENTINEL_I16

    header = struct.pack("<BBII", 2, 1, width, height)
    header += struct.pack("<4d", *extent_3857)
    header += struct.pack("<d", quantum)
    compressed = zlib.compress(np.ascontiguousarray(quant, dtype="<i2").tobytes(), level=9)
    payload = header + compressed

    size = len(payload)
    chunk = fourcc + struct.pack("<I", size) + payload
    if size % 2 == 1:
        chunk += b"\x00"  # RIFF-Padding auf gerade Länge, zählt nicht zu size

    with open(webp_path, "rb") as f:
        content = f.read()

    if content[0:4] != b"RIFF" or content[8:12] != b"WEBP":
        raise ValueError(f"{webp_path} ist keine gültige WebP-Datei (RIFF/WEBP-Header fehlt)")

    riff_size = struct.unpack("<I", content[4:8])[0]
    new_riff_size = riff_size + len(chunk)

    with open(webp_path, "wb") as f:
        f.write(content[:4])
        f.write(struct.pack("<I", new_riff_size))
        f.write(content[8:])
        f.write(chunk)


# ------------------------------
# Dateien durchgehen
# ------------------------------
all_files_global = sorted([f for f in os.listdir(data_dir) if f.endswith(".grib2")])

for filename in all_files_global:
    path = os.path.join(data_dir, filename)
    ds = cfgrib.open_dataset(path)

    valid_time_utc_override = None

    # ------------------------------
    # Daten je Typ (Logik unverändert aus dem Original übernommen)
    # ------------------------------
    if var_type == "t2m":
        if "t2m" not in ds:
            print(f"Keine t2m in {filename}")
            ds.close()
            continue
        data = ds["t2m"].values - 273.15
        cmap, norm = t2m_colors, t2m_norm
    elif var_type == "t850":
        if "t" not in ds:
            print(f"Keine t in {filename} ds.keys(): {list(ds.keys())}")
            ds.close()
            continue
        data = ds["t"].values - 273.15
        cmap, norm = t2m_colors, t2m_norm
    elif var_type == "ww":
        varname = next((vn for vn in ds.data_vars if vn.lower() in ["ww", "weather"]), None)
        if varname is None:
            print(f"Keine WW in {filename}")
            ds.close()
            continue
        data = ds[varname].values
        cmap = None  # wird nach dem Codieren dynamisch gesetzt
    elif var_type == "tp":
        if "tp" not in ds:
            print(f"Keine tp in {filename}")
            ds.close()
            continue

        idx_now = all_files_global.index(filename)
        if idx_now >= 78:
            print(f"Datei {filename}: Überspringe stündlichen Niederschlag (3h Schritte)")
            ds.close()
            continue
        if idx_now + 1 >= len(all_files_global):
            print(f"{filename}: keine folgende Datei -> 1h-Niederschlag nicht berechenbar, überspringe")
            ds.close()
            continue

        next_path = os.path.join(data_dir, all_files_global[idx_now + 1])
        ds_next = cfgrib.open_dataset(next_path)

        if "tp" not in ds_next:
            print(f"Keine tp in {all_files_global[idx_now + 1]}")
            ds_next.close()
            ds.close()
            continue

        tp_now = ds["tp"].values
        tp_next = ds_next["tp"].values
        data = tp_next - tp_now
        data[data < 0.1] = np.nan

        vt_next_raw = ds_next["valid_time"].values
        valid_time_utc_override = pd.to_datetime(vt_next_raw[0]) if np.ndim(vt_next_raw) > 0 else pd.to_datetime(vt_next_raw)
        ds_next.close()
        cmap, norm = prec_colors, prec_norm
    elif var_type == "tp_acc":
        tp_var = next((vn for vn in ["tp", "tot_prec"] if vn in ds), None)
        if tp_var is None:
            print(f"Keine Niederschlagsvariable in {filename}")
            ds.close()
            continue
        lon_tmp = ds["longitude"].values
        lat_tmp = ds["latitude"].values
        tp_all = ds[tp_var].values
        if tp_all.ndim == 1:
            ny, nx = len(lat_tmp), len(lon_tmp)
            data = tp_all.reshape(ny, nx)
        elif tp_all.ndim == 3:
            data = tp_all[3] - tp_all[0] if tp_all.shape[0] > 1 else tp_all[0]
        else:
            data = tp_all
        cmap, norm = tp_acc_colors, tp_acc_norm
    elif var_type == "dbz_cmax":
        if "DBZ_CMAX" not in ds:
            print(f"Keine DBZ_CMAX in {filename} ds.keys(): {list(ds.keys())}")
            ds.close()
            continue
        data = ds["DBZ_CMAX"].values[0, :, :]
        cmap, norm = dbz_colors, dbz_norm
    elif var_type == "wind":
        if "fg10" not in ds:
            print(f"Keine passende Windvariable in {filename} ds.keys(): {list(ds.keys())}")
            ds.close()
            continue
        data = ds["fg10"].values
        data[data < 0] = np.nan
        data = data * 3.6  # m/s -> km/h
        cmap, norm = wind_colors, wind_norm
    elif var_type == "snow":
        if "sde" not in ds:
            print(f"Keine sde-Variable in {filename}")
            ds.close()
            continue
        data = ds["sde"].values
        data[data < 0] = np.nan
        data = data * 100  # -> cm
        cmap, norm = snow_colors, snow_norm
    elif var_type == "twater":
        if "TWATER" not in ds:
            print(f"Keine TWATER-Variable in {filename}")
            ds.close()
            continue
        data = ds["TWATER"].values
        data[data < 0] = np.nan
        cmap, norm = twater_colors, twater_norm
    elif var_type == "snowfall":
        if "SNOWLMT" not in ds:
            print(f"Keine SNOWLMT-Variable in {filename} ds.keys(): {list(ds.keys())}")
            ds.close()
            continue
        data = ds["SNOWLMT"].values
        data[data < 0] = np.nan
        cmap, norm = snowfall_colors, snowfall_norm
    elif var_type == "pmsl":
        if "prmsl" not in ds:
            print(f"Keine prmsl-Variable in {filename} ds.keys(): {list(ds.keys())}")
            ds.close()
            continue
        data = ds["prmsl"].values / 100
        data[data < 0] = np.nan
        cmap, norm = pmsl_colors, pmsl_norm
    else:
        print(f"Unbekannter var_type {var_type}")
        ds.close()
        continue

    if data.ndim == 3:
        data = data[0]

    lon = ds["longitude"].values
    lat = ds["latitude"].values
    run_time_utc = pd.to_datetime(ds["time"].values) if "time" in ds else None

    if valid_time_utc_override is not None:
        valid_time_utc = valid_time_utc_override
    elif "valid_time" in ds:
        valid_time_raw = ds["valid_time"].values
        valid_time_utc = pd.to_datetime(valid_time_raw[0]) if np.ndim(valid_time_raw) > 0 else pd.to_datetime(valid_time_raw)
    else:
        step = pd.to_timedelta(ds["step"].values[0])
        valid_time_utc = run_time_utc + step
    valid_time_local = valid_time_utc.tz_localize("UTC").astimezone(ZoneInfo("Europe/Berlin"))

    # ---------------------------------
    # Natives Modellgitter beibehalten (keine Interpolation auf ein
    # feineres/anderes Gitter mehr) - lediglich sicherstellen, dass lat/lon
    # aufsteigend sortiert sind, da RegularGridInterpolator (in
    # warp_equirect_to_webmercator) das voraussetzt.
    # ---------------------------------
    if lon.ndim == 1 and lat.ndim == 1 and data.ndim == 2:
        if lat[0] > lat[-1]:
            lat = lat[::-1]
            data = data[::-1, :]
        if lon[0] > lon[-1]:
            lon = lon[::-1]
            data = data[:, ::-1]

    # data ist jetzt (lat aufsteigend, lon aufsteigend) sortiert,
    # Zeile 0 = Süden. Für save_transparent_webp reicht das - dort wird
    # zum Speichern vertikal gespiegelt (Bildzeile 0 = Norden).

    # ------------------------------
    # WW-Farbcodierung (Codes -> fortlaufender Index)
    # ------------------------------
    if var_type == "ww":
        valid_mask = np.isfinite(data)
        codes = np.unique(data[valid_mask]).astype(int)
        codes = [c for c in codes if c in ww_colors_base and c not in ignore_codes]
        codes.sort()
        cmap = ListedColormap([ww_colors_base[c] for c in codes]) if codes else ListedColormap(["#FFFFFF00"])
        norm = mcolors.Normalize(vmin=-0.5, vmax=max(len(codes) - 0.5, 0.5))
        code2idx = {c: i for i, c in enumerate(codes)}
        idx_data = np.full_like(data, fill_value=np.nan, dtype=float)
        for c, i in code2idx.items():
            idx_data[data == c] = i
        render_data = idx_data
    else:
        render_data = data

    # ------------------------------
    # Nach EPSG:3857 (Web Mercator) umprojizieren
    # ------------------------------
    merc_method = "nearest" if var_type == "ww" else "linear"
    render_data_merc = warp_equirect_to_webmercator(
        render_data, lon, lat, extent, method=merc_method
    )

    # ------------------------------
    # Isobaren (pmsl) als Linien-Layer ins Bild backen
    # ------------------------------
    contour_rgba = None
    if var_type == "pmsl":
        x_new_m, y_new_m = webmercator_target_grid(extent)
        contour_rgba = render_contour_overlay_rgba(
            x_new_m, y_new_m, render_data_merc,
            main_levels=list(range(912, 1070, 4)),
            fine_levels=list(range(912, 1070, 1)),
        )

    # ------------------------------
    # Transparentes WebP speichern
    # ------------------------------
    outname = f"{var_type}_{valid_time_local:%Y%m%d_%H%M}.webp"
    out_path = os.path.join(output_dir, outname)
    save_transparent_webp(render_data_merc, cmap, norm, out_path, contour_rgba=contour_rgba)

    # Für t2m/wind zusätzlich die echten physikalischen Werte (°C bzw.
    # km/h, nicht die Farben) als privaten RIFF-Chunk direkt ins WebP
    # einbetten - row0 = Norden, damit der Chunk 1:1 zur Bildorientierung
    # passt (das Bild wird in save_transparent_webp beim Speichern
    # gespiegelt, render_data_merc selbst hat row0 = Süden).
    if var_type in EMBED_DATA_VARS:
        germany_data = crop_to_germany(render_data_merc)          # row0 = Süden
        quantum = QUANTUM_STEP.get(var_type, 0.1)
        embed_data_chunk(out_path, germany_data[::-1], GERMANY_CROP_EXTENT_3857, quantum)  # row0 = Norden

    print(f"{filename} -> {outname}")

    # ------------------------------
    # Aufräumen - wichtig bei vielen Dateien in der Schleife!
    # ------------------------------
    ds.close()
    del data, render_data, render_data_merc
    gc.collect()
