"""Generate an interactive SVG viewer for an OSMnx GraphML road network.

The output is a standalone HTML file: no web server, Python package, or network
access is required once it has been generated.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPHML = PROJECT_ROOT / "datasets" / "small_boston.graphml"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "small_boston_interactive.html"
GRAPHML_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}
LINESTRING_RE = re.compile(r"LINESTRING\s*\((.*?)\)", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


@dataclass
class Node:
    id: str
    lon: float
    lat: float
    attrs: dict[str, str]
    x: float = 0.0
    y: float = 0.0


@dataclass
class Edge:
    id: str
    source: str
    target: str
    attrs: dict[str, str]
    points: list[tuple[float, float]]
    svg_points: list[tuple[float, float]]
    label_x: float
    label_y: float
    weight: str


def parse_graphml(path: Path) -> tuple[dict[str, Node], list[dict[str, Any]], bool]:
    tree = ET.parse(path)
    root = tree.getroot()

    keys: dict[str, str] = {}
    for key in root.findall("g:key", GRAPHML_NS):
        key_id = key.attrib.get("id")
        attr_name = key.attrib.get("attr.name", key_id)
        if key_id:
            keys[key_id] = attr_name or key_id

    graph = root.find("g:graph", GRAPHML_NS)
    if graph is None:
        raise ValueError(f"No <graph> element found in {path}")
    directed = graph.attrib.get("edgedefault") == "directed"

    nodes: dict[str, Node] = {}
    for node_elem in graph.findall("g:node", GRAPHML_NS):
        node_id = node_elem.attrib["id"]
        attrs = read_data(node_elem, keys)
        lon = first_float(attrs, ("x", "lon", "longitude"))
        lat = first_float(attrs, ("y", "lat", "latitude"))
        if lon is None or lat is None:
            raise ValueError(f"Node {node_id!r} is missing x/y or lon/lat coordinates")
        nodes[node_id] = Node(node_id, lon, lat, attrs)

    raw_edges: list[dict[str, Any]] = []
    for index, edge_elem in enumerate(graph.findall("g:edge", GRAPHML_NS)):
        source = edge_elem.attrib["source"]
        target = edge_elem.attrib["target"]
        attrs = read_data(edge_elem, keys)
        edge_id = edge_elem.attrib.get("id", str(index))
        raw_edges.append(
            {
                "id": f"{source}->{target}#{edge_id}:{index}",
                "source": source,
                "target": target,
                "attrs": attrs,
            }
        )

    return nodes, raw_edges, directed


def read_data(elem: ET.Element, keys: dict[str, str]) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for data in elem.findall("g:data", GRAPHML_NS):
        key = data.attrib.get("key", "")
        name = keys.get(key, key)
        attrs[name] = data.text or ""
    return attrs


def first_float(attrs: dict[str, str], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name not in attrs:
            continue
        try:
            return float(attrs[name])
        except ValueError:
            continue
    return None


def parse_linestring(value: str) -> list[tuple[float, float]]:
    match = LINESTRING_RE.search(value or "")
    if not match:
        return []
    numbers = [float(number) for number in NUMBER_RE.findall(match.group(1))]
    return list(zip(numbers[0::2], numbers[1::2]))


def project_nodes(nodes: dict[str, Node]) -> dict[str, float]:
    lons = [node.lon for node in nodes.values()]
    lats = [node.lat for node in nodes.values()]
    center_lat = math.radians(sum(lats) / len(lats))
    lon_scale = math.cos(center_lat)

    projected = [(node.lon * lon_scale, node.lat) for node in nodes.values()]
    min_x = min(x for x, _ in projected)
    max_x = max(x for x, _ in projected)
    min_y = min(y for _, y in projected)
    max_y = max(y for _, y in projected)

    scale = 100_000.0
    for node in nodes.values():
        node.x = (node.lon * lon_scale - min_x) * scale
        node.y = (max_y - node.lat) * scale

    return {
        "min_lon": min(lons),
        "max_lon": max(lons),
        "min_lat": min(lats),
        "max_lat": max(lats),
        "width": max((max_x - min_x) * scale, 1.0),
        "height": max((max_y - min_y) * scale, 1.0),
        "center_lat": sum(lats) / len(lats),
        "center_lon": sum(lons) / len(lons),
        "lon_scale": lon_scale,
        "min_x": min_x,
        "max_y": max_y,
        "scale": scale,
    }


def project_point(lon: float, lat: float, bounds: dict[str, float]) -> tuple[float, float]:
    x = (lon * bounds["lon_scale"] - bounds["min_x"]) * bounds["scale"]
    y = (bounds["max_y"] - lat) * bounds["scale"]
    return x, y


def build_edges(
    raw_edges: list[dict[str, Any]],
    nodes: dict[str, Node],
    bounds: dict[str, float],
    weight_attr: str,
) -> list[Edge]:
    edges: list[Edge] = []
    for raw in raw_edges:
        source = raw["source"]
        target = raw["target"]
        attrs = raw["attrs"]
        geometry = parse_linestring(attrs.get("geometry", ""))
        if not geometry:
            geometry = [(nodes[source].lon, nodes[source].lat), (nodes[target].lon, nodes[target].lat)]

        svg_points = [project_point(lon, lat, bounds) for lon, lat in geometry]
        label_x, label_y = midpoint(svg_points)
        weight_value = attrs.get(weight_attr, attrs.get("length", ""))
        weight = format_weight(weight_value)
        edges.append(
            Edge(
                id=raw["id"],
                source=source,
                target=target,
                attrs=attrs,
                points=geometry,
                svg_points=svg_points,
                label_x=label_x,
                label_y=label_y,
                weight=weight,
            )
        )
    return edges


def midpoint(points: list[tuple[float, float]]) -> tuple[float, float]:
    if len(points) == 1:
        return points[0]

    lengths: list[float] = []
    total = 0.0
    for a, b in zip(points, points[1:]):
        length = math.dist(a, b)
        lengths.append(length)
        total += length

    half = total / 2.0
    walked = 0.0
    for index, length in enumerate(lengths):
        if walked + length >= half and length > 0:
            a = points[index]
            b = points[index + 1]
            ratio = (half - walked) / length
            return a[0] + (b[0] - a[0]) * ratio, a[1] + (b[1] - a[1]) * ratio
        walked += length
    return points[len(points) // 2]


def format_weight(value: str) -> str:
    try:
        return f"{float(value):.1f} m"
    except (TypeError, ValueError):
        return str(value)


def graph_payload(
    path: Path,
    nodes: dict[str, Node],
    edges: list[Edge],
    bounds: dict[str, float],
    directed: bool,
    weight_attr: str,
) -> dict[str, Any]:
    degree: dict[str, dict[str, int]] = {node_id: {"in": 0, "out": 0} for node_id in nodes}
    for edge in edges:
        degree[edge.source]["out"] += 1
        degree[edge.target]["in"] += 1

    return {
        "graph": {
            "path": str(path),
            "directed": directed,
            "weightAttr": weight_attr,
            "nodeCount": len(nodes),
            "edgeCount": len(edges),
            "bounds": {
                "minLon": bounds["min_lon"],
                "maxLon": bounds["max_lon"],
                "minLat": bounds["min_lat"],
                "maxLat": bounds["max_lat"],
                "centerLon": bounds["center_lon"],
                "centerLat": bounds["center_lat"],
                "width": bounds["width"],
                "height": bounds["height"],
            },
        },
        "nodes": [
            {
                "id": node.id,
                "lon": node.lon,
                "lat": node.lat,
                "x": round(node.x, 4),
                "y": round(node.y, 4),
                "attrs": node.attrs,
                "inDegree": degree[node.id]["in"],
                "outDegree": degree[node.id]["out"],
            }
            for node in nodes.values()
        ],
        "edges": [
            {
                "id": edge.id,
                "source": edge.source,
                "target": edge.target,
                "weight": edge.weight,
                "labelX": round(edge.label_x, 4),
                "labelY": round(edge.label_y, 4),
                "points": [[round(x, 4), round(y, 4)] for x, y in edge.svg_points],
                "attrs": edge.attrs,
            }
            for edge in edges
        ],
    }


def write_html(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    title = f"GraphML viewer - {Path(payload['graph']['path']).name}"
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    output.write_text(render_html(title, data), encoding="utf-8")


def render_html(title: str, data_json: str) -> str:
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #64717f;
      --line: #cfd7df;
      --road: #64748b;
      --road-hi: #f97316;
      --node: #0f766e;
      --node-hi: #be123c;
      --label: #26384d;
      --accent: #2563eb;
      --shadow: 0 16px 45px rgba(15, 23, 42, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ height: 100%; }}
    body {{
      margin: 0;
      overflow: hidden;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
    }}
    .app {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      height: 100vh;
    }}
    .map-wrap {{
      position: relative;
      min-width: 0;
      background:
        linear-gradient(rgba(100, 116, 139, 0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(100, 116, 139, 0.06) 1px, transparent 1px),
        #fbfcfd;
      background-size: 36px 36px;
    }}
    svg {{
      width: 100%;
      height: 100%;
      display: block;
      cursor: grab;
      touch-action: none;
      user-select: none;
    }}
    svg.dragging {{ cursor: grabbing; }}
    svg.draw-mode {{ cursor: crosshair; }}
    svg.draw-mode .node,
    svg.draw-mode .edge,
    svg.draw-mode .edge.hit {{ pointer-events: none; }}
    .toolbar {{
      position: absolute;
      top: 14px;
      left: 14px;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.94);
      box-shadow: var(--shadow);
    }}
    button, input {{
      height: 32px;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      font: inherit;
      font-size: 13px;
    }}
    button {{
      min-width: 34px;
      padding: 0 10px;
      cursor: pointer;
      white-space: nowrap;
    }}
    button.active {{
      border-color: var(--accent);
      color: var(--accent);
      background: #eff6ff;
    }}
    input {{
      width: 168px;
      padding: 0 10px;
      outline: none;
    }}
    input:focus {{ border-color: var(--accent); }}
    textarea {{
      width: 100%;
      min-height: 170px;
      padding: 10px;
      border: 1px solid var(--line);
      resize: vertical;
      color: var(--ink);
      font: 13px/1.45 Consolas, "Courier New", monospace;
      outline: none;
    }}
    textarea:focus {{ border-color: var(--accent); }}
    .palette {{
      display: flex;
      align-items: center;
      gap: 5px;
      padding: 0 2px;
    }}
    .color-swatch {{
      width: 24px;
      min-width: 24px;
      height: 24px;
      padding: 0;
      border-radius: 50%;
      border: 2px solid white;
      outline: 1px solid var(--line);
      background: var(--swatch);
    }}
    .color-swatch.active {{
      outline: 2px solid var(--accent);
      border-color: white;
      background: var(--swatch);
    }}
    .side {{
      min-width: 0;
      overflow: auto;
      border-left: 1px solid var(--line);
      background: var(--panel);
    }}
    .side header {{
      position: sticky;
      top: 0;
      z-index: 2;
      padding: 18px 18px 14px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.96);
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      line-height: 1.25;
      letter-spacing: 0;
    }}
    .path {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .section {{
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
    }}
    .section h2 {{
      margin: 0 0 10px;
      font-size: 13px;
      line-height: 1.2;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    dl {{
      display: grid;
      grid-template-columns: 92px minmax(0, 1fr);
      gap: 8px 12px;
      margin: 0;
      font-size: 13px;
      line-height: 1.35;
    }}
    dt {{ color: var(--muted); }}
    dd {{
      margin: 0;
      min-width: 0;
      overflow-wrap: anywhere;
    }}
    .empty {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .edge {{
      fill: none;
      stroke: var(--road);
      stroke-width: 2.1;
      stroke-linecap: round;
      stroke-linejoin: round;
      vector-effect: non-scaling-stroke;
      opacity: 0.74;
      pointer-events: none;
    }}
    .edge.hit {{
      stroke: transparent;
      stroke-width: 10;
      vector-effect: non-scaling-stroke;
      opacity: 0;
      cursor: pointer;
      pointer-events: stroke;
    }}
    .edge.highlight {{
      stroke: var(--road-hi);
      stroke-width: 3.8;
      opacity: 1;
    }}
    .edge.dim {{ opacity: 0.18; }}
    .node {{
      fill: rgba(255, 255, 255, 0.68);
      stroke: var(--node);
      stroke-width: 1.7;
      vector-effect: non-scaling-stroke;
      cursor: pointer;
    }}
    .node.highlight {{
      fill: var(--node-hi);
      stroke: white;
      stroke-width: 2.7;
    }}
    .node.dim {{ opacity: 0.3; }}
    .node-label, .edge-label {{
      pointer-events: none;
      paint-order: stroke;
      stroke: rgba(255, 255, 255, 0.9);
      stroke-width: 4px;
      stroke-linejoin: round;
      fill: var(--label);
      font-family: Arial, Helvetica, sans-serif;
      letter-spacing: 0;
    }}
    .node-label {{ font-size: 13px; font-weight: 700; }}
    .edge-label {{ font-size: 11px; fill: #7c2d12; }}
    .labels-off .node-label, .edge-labels-off .edge-label {{ display: none; }}
    .sequence-marker {{
      pointer-events: none;
    }}
    .sequence-marker line {{
      stroke: rgba(17, 24, 39, 0.52);
      stroke-width: 1.2;
      vector-effect: non-scaling-stroke;
    }}
    .sequence-marker circle {{
      fill: rgba(250, 204, 21, 0.84);
      stroke: #111827;
      stroke-width: 1.4;
      vector-effect: non-scaling-stroke;
    }}
    .sequence-marker text {{
      fill: #111827;
      font-family: Arial, Helvetica, sans-serif;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0;
      text-anchor: middle;
      dominant-baseline: central;
      paint-order: stroke;
      stroke: rgba(255, 255, 255, 0.88);
      stroke-width: 2px;
      stroke-linejoin: round;
    }}
    .drawing-stroke {{
      fill: none;
      stroke-width: 4.5;
      stroke-linecap: round;
      stroke-linejoin: round;
      vector-effect: non-scaling-stroke;
      pointer-events: none;
      opacity: 0.92;
    }}
    .tooltip {{
      position: absolute;
      z-index: 4;
      max-width: 260px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      font-size: 12px;
      line-height: 1.35;
      box-shadow: var(--shadow);
      pointer-events: none;
      opacity: 0;
      transform: translate(12px, 12px);
      transition: opacity 90ms ease;
    }}
    .tooltip.show {{ opacity: 1; }}
    .mini {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .metric {{
      border: 1px solid var(--line);
      padding: 10px;
      background: #fbfcfd;
    }}
    .metric strong {{
      display: block;
      font-size: 18px;
      line-height: 1;
    }}
    .metric span {{
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }}
    .note {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }}
    dialog {{
      width: min(560px, calc(100vw - 32px));
      border: 1px solid var(--line);
      padding: 0;
      background: white;
      color: var(--ink);
      box-shadow: var(--shadow);
    }}
    dialog::backdrop {{
      background: rgba(15, 23, 42, 0.28);
    }}
    .dialog-shell {{
      padding: 16px;
    }}
    .dialog-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 12px;
    }}
    .dialog-head h2 {{
      margin: 0;
      font-size: 16px;
      line-height: 1.25;
      letter-spacing: 0;
    }}
    .dialog-actions {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-top: 12px;
    }}
    .dialog-actions .left {{
      display: flex;
      gap: 8px;
    }}
    #sequenceStatus {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      text-align: right;
    }}
    @media (max-width: 900px) {{
      .app {{ grid-template-columns: 1fr; grid-template-rows: minmax(0, 1fr) 270px; }}
      .side {{ border-left: 0; border-top: 1px solid var(--line); }}
      .toolbar {{ right: 10px; left: 10px; flex-wrap: wrap; }}
      input {{ flex: 1; min-width: 150px; width: auto; }}
    }}
  </style>
</head>
<body>
  <main class="app">
    <section class="map-wrap">
      <div class="toolbar" aria-label="Map controls">
        <button id="zoomIn" title="Zoom in">+</button>
        <button id="zoomOut" title="Zoom out">-</button>
        <button id="resetView" title="Fit graph">Fit</button>
        <button id="toggleNodeLabels" class="active" title="Toggle node ids">IDs</button>
        <button id="toggleEdgeLabels" class="active" title="Toggle edge weights">Weights</button>
        <button id="openSequenceDialog" title="Paste node ids and show sequence labels">Seq</button>
        <button id="toggleDraw" title="Toggle drawing mode">Draw</button>
        <div class="palette" aria-label="Drawing colors">
          <button class="color-swatch active" data-color="#ef4444" style="--swatch:#ef4444" title="Red"></button>
          <button class="color-swatch" data-color="#2563eb" style="--swatch:#2563eb" title="Blue"></button>
          <button class="color-swatch" data-color="#16a34a" style="--swatch:#16a34a" title="Green"></button>
          <button class="color-swatch" data-color="#f59e0b" style="--swatch:#f59e0b" title="Amber"></button>
          <button class="color-swatch" data-color="#111827" style="--swatch:#111827" title="Black"></button>
        </div>
        <button id="undoDrawing" title="Undo the last drawing stroke">Undo</button>
        <button id="clearDrawings" title="Clear all drawing strokes">Clear</button>
        <input id="nodeSearch" list="nodeIds" placeholder="Node id">
        <datalist id="nodeIds"></datalist>
      </div>
      <svg id="graphSvg" role="img" aria-label="Interactive GraphML road network">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"></path>
          </marker>
          <marker id="arrowHi" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#f97316"></path>
          </marker>
        </defs>
        <g id="viewport">
          <g id="edgeHitLayer"></g>
          <g id="nodeLayer"></g>
          <g id="edgeLayer"></g>
          <g id="edgeLabelLayer"></g>
          <g id="nodeLabelLayer"></g>
          <g id="drawingLayer"></g>
          <g id="sequenceLayer"></g>
        </g>
      </svg>
      <div id="tooltip" class="tooltip"></div>
      <dialog id="sequenceDialog">
        <div class="dialog-shell">
          <div class="dialog-head">
            <h2>Node sequence labels</h2>
            <button id="closeSequenceDialog" type="button">Close</button>
          </div>
          <textarea id="sequenceInput" spellcheck="false" placeholder="Paste node ids here, for example: ['100522728', '100522741', '1061531429']"></textarea>
          <div class="dialog-actions">
            <div class="left">
              <button id="applySequence" type="button">Apply</button>
              <button id="clearSequence" type="button">Clear</button>
            </div>
            <div id="sequenceStatus">No sequence labels.</div>
          </div>
        </div>
      </dialog>
    </section>
    <aside class="side">
      <header>
        <h1>{safe_title}</h1>
        <div id="graphPath" class="path"></div>
      </header>
      <section class="section">
        <h2>Summary</h2>
        <div class="mini">
          <div class="metric"><strong id="nodeCount"></strong><span>Nodes</span></div>
          <div class="metric"><strong id="edgeCount"></strong><span>Edges</span></div>
          <div class="metric"><strong id="directed"></strong><span>Directed</span></div>
          <div class="metric"><strong id="weightAttr"></strong><span>Weight</span></div>
        </div>
      </section>
      <section class="section">
        <h2>Selection</h2>
        <div id="selection" class="empty">Click a node to inspect longitude and latitude.</div>
      </section>
      <section class="section">
        <h2>Sequence</h2>
        <div id="sequenceInfo" class="empty">Paste node ids with Seq to mark ordered labels from 0.</div>
      </section>
      <section class="section">
        <h2>Drawing</h2>
        <div id="drawingInfo" class="empty">Turn on Draw to sketch over the graph.</div>
      </section>
      <section class="section">
        <h2>Bounds</h2>
        <dl id="bounds"></dl>
      </section>
    </aside>
  </main>
  <script id="graphData" type="application/json">{data_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById("graphData").textContent);
    const svg = document.getElementById("graphSvg");
    const viewport = document.getElementById("viewport");
    const edgeLayer = document.getElementById("edgeLayer");
    const edgeHitLayer = document.getElementById("edgeHitLayer");
    const edgeLabelLayer = document.getElementById("edgeLabelLayer");
    const nodeLayer = document.getElementById("nodeLayer");
    const nodeLabelLayer = document.getElementById("nodeLabelLayer");
    const drawingLayer = document.getElementById("drawingLayer");
    const sequenceLayer = document.getElementById("sequenceLayer");
    const tooltip = document.getElementById("tooltip");
    const selection = document.getElementById("selection");
    const sequenceDialog = document.getElementById("sequenceDialog");
    const sequenceInput = document.getElementById("sequenceInput");
    const sequenceStatus = document.getElementById("sequenceStatus");
    const sequenceInfo = document.getElementById("sequenceInfo");
    const drawingInfo = document.getElementById("drawingInfo");
    const toggleDrawButton = document.getElementById("toggleDraw");
    const colorButtons = Array.from(document.querySelectorAll(".color-swatch"));
    const nodeById = new Map(payload.nodes.map(node => [node.id, node]));
    const edgeById = new Map(payload.edges.map(edge => [edge.id, edge]));
    const incidentEdgeIds = new Map(payload.nodes.map(node => [node.id, []]));
    const edgeElements = new Map();
    const edgeHitElements = new Map();
    const nodeElements = new Map();
    const sequenceIndexByNode = new Map();
    const drawings = [];
    const graphBounds = payload.graph.bounds;
    const pad = 70;
    const defaultDrawingColor = "#ef4444";
    let viewBox = {{
      x: -pad,
      y: -pad,
      w: graphBounds.width + pad * 2,
      h: graphBounds.height + pad * 2
    }};
    let isDragging = false;
    let dragStart = null;
    let selectedNodeId = null;
    let selectedEdgeId = null;
    let isDrawMode = false;
    let activeDrawingColor = defaultDrawingColor;
    let activeStroke = null;

    for (const edge of payload.edges) {{
      if (incidentEdgeIds.has(edge.source)) incidentEdgeIds.get(edge.source).push(edge.id);
      if (incidentEdgeIds.has(edge.target)) incidentEdgeIds.get(edge.target).push(edge.id);
    }}

    initSummary();
    renderGraph();
    fitView();
    updateDrawingInfo();

    document.getElementById("zoomIn").addEventListener("click", () => zoomAtCenter(0.75));
    document.getElementById("zoomOut").addEventListener("click", () => zoomAtCenter(1.35));
    document.getElementById("resetView").addEventListener("click", fitView);
    document.getElementById("toggleNodeLabels").addEventListener("click", event => {{
      event.currentTarget.classList.toggle("active");
      svg.classList.toggle("labels-off");
    }});
    document.getElementById("toggleEdgeLabels").addEventListener("click", event => {{
      event.currentTarget.classList.toggle("active");
      svg.classList.toggle("edge-labels-off");
    }});
    document.getElementById("openSequenceDialog").addEventListener("click", () => {{
      openSequenceDialog();
    }});
    document.getElementById("closeSequenceDialog").addEventListener("click", () => sequenceDialog.close());
    document.getElementById("applySequence").addEventListener("click", updateSequenceMarkers);
    document.getElementById("clearSequence").addEventListener("click", () => {{
      sequenceInput.value = "";
      updateSequenceMarkers();
    }});
    sequenceInput.addEventListener("input", updateSequenceMarkers);
    toggleDrawButton.addEventListener("click", () => setDrawMode(!isDrawMode));
    for (const button of colorButtons) {{
      button.addEventListener("click", () => {{
        activeDrawingColor = button.dataset.color || defaultDrawingColor;
        for (const item of colorButtons) item.classList.toggle("active", item === button);
        updateDrawingInfo();
      }});
    }}
    document.getElementById("undoDrawing").addEventListener("click", undoDrawing);
    document.getElementById("clearDrawings").addEventListener("click", clearDrawings);
    document.getElementById("nodeSearch").addEventListener("change", event => {{
      const value = event.currentTarget.value.trim();
      if (nodeById.has(value)) selectNode(value, true);
    }});

    svg.addEventListener("wheel", event => {{
      event.preventDefault();
      const factor = event.deltaY < 0 ? 0.82 : 1.22;
      const point = screenToSvg(event.clientX, event.clientY);
      zoomAt(point.x, point.y, factor);
    }}, {{ passive: false }});

    svg.addEventListener("pointerdown", event => {{
      if (isDrawMode) {{
        beginDrawing(event);
        return;
      }}
      if (event.target.closest(".node") || event.target.closest(".edge")) return;
      isDragging = true;
      svg.classList.add("dragging");
      svg.setPointerCapture(event.pointerId);
      dragStart = {{ clientX: event.clientX, clientY: event.clientY, viewBox: {{ ...viewBox }} }};
    }});
    svg.addEventListener("pointermove", event => {{
      if (activeStroke) {{
        extendDrawing(event);
        return;
      }}
      if (!isDragging || !dragStart) return;
      const scaleX = viewBox.w / svg.clientWidth;
      const scaleY = viewBox.h / svg.clientHeight;
      viewBox.x = dragStart.viewBox.x - (event.clientX - dragStart.clientX) * scaleX;
      viewBox.y = dragStart.viewBox.y - (event.clientY - dragStart.clientY) * scaleY;
      applyViewBox();
    }});
    svg.addEventListener("pointerup", event => {{
      if (activeStroke) {{
        finishDrawing(event);
        return;
      }}
      stopDrag(event);
    }});
    svg.addEventListener("pointercancel", event => {{
      if (activeStroke) {{
        finishDrawing(event);
        return;
      }}
      stopDrag(event);
    }});
    svg.addEventListener("mouseleave", () => hideTooltip());

    function initSummary() {{
      document.getElementById("graphPath").textContent = payload.graph.path;
      document.getElementById("nodeCount").textContent = payload.graph.nodeCount;
      document.getElementById("edgeCount").textContent = payload.graph.edgeCount;
      document.getElementById("directed").textContent = payload.graph.directed ? "Yes" : "No";
      document.getElementById("weightAttr").textContent = payload.graph.weightAttr;
      document.getElementById("bounds").innerHTML = [
        ["Min lon", graphBounds.minLon.toFixed(7)],
        ["Max lon", graphBounds.maxLon.toFixed(7)],
        ["Min lat", graphBounds.minLat.toFixed(7)],
        ["Max lat", graphBounds.maxLat.toFixed(7)]
      ].map(([key, value]) => `<dt>${{key}}</dt><dd>${{value}}</dd>`).join("");

      const datalist = document.getElementById("nodeIds");
      datalist.innerHTML = payload.nodes.map(node => `<option value="${{escapeAttr(node.id)}}"></option>`).join("");
    }}

    function renderGraph() {{
      for (const edge of payload.edges) {{
        const pathData = edgePath(edge.points);
        const path = createSvg("path", {{
          d: pathData,
          class: "edge",
          "data-edge-id": edge.id,
          "marker-end": payload.graph.directed ? "url(#arrow)" : ""
        }});
        path.addEventListener("mouseenter", event => {{
          highlightEdge(edge.id);
          showTooltip(event, edgeTooltip(edge));
        }});
        path.addEventListener("mousemove", event => showTooltip(event, edgeTooltip(edge)));
        path.addEventListener("mouseleave", () => restoreHighlight());
        path.addEventListener("click", event => {{
          event.stopPropagation();
          selectEdge(edge.id);
        }});
        edgeLayer.appendChild(path);
        edgeElements.set(edge.id, path);

        const hit = createSvg("path", {{
          d: pathData,
          class: "edge hit",
          "data-edge-id": edge.id
        }});
        hit.addEventListener("mouseenter", event => {{
          highlightEdge(edge.id);
          showTooltip(event, edgeTooltip(edge));
        }});
        hit.addEventListener("mousemove", event => showTooltip(event, edgeTooltip(edge)));
        hit.addEventListener("mouseleave", () => restoreHighlight());
        hit.addEventListener("click", event => {{
          event.stopPropagation();
          selectEdge(edge.id);
        }});
        edgeHitLayer.appendChild(hit);
        edgeHitElements.set(edge.id, hit);

        const label = createSvg("text", {{
          x: edge.labelX,
          y: edge.labelY,
          class: "edge-label",
          "text-anchor": "middle",
          "dominant-baseline": "central"
        }});
        label.textContent = edge.weight;
        edgeLabelLayer.appendChild(label);
      }}

      for (const node of payload.nodes) {{
        const circle = createSvg("circle", {{
          cx: node.x,
          cy: node.y,
          r: 3.4,
          class: "node",
          "data-node-id": node.id
        }});
        circle.addEventListener("mouseenter", event => {{
          highlightNode(node.id);
          showTooltip(event, nodeTooltip(node));
        }});
        circle.addEventListener("mousemove", event => showTooltip(event, nodeTooltip(node)));
        circle.addEventListener("mouseleave", () => restoreHighlight());
        circle.addEventListener("click", event => {{
          event.stopPropagation();
          selectNode(node.id, false);
        }});
        nodeLayer.appendChild(circle);
        nodeElements.set(node.id, circle);

        const label = createSvg("text", {{
          x: node.x + 7,
          y: node.y - 7,
          class: "node-label"
        }});
        label.textContent = node.id;
        nodeLabelLayer.appendChild(label);
      }}
      svg.addEventListener("click", () => {{
        if (!isDrawMode) clearSelection();
      }});
    }}

    function edgePath(points) {{
      return points.map((point, index) => `${{index === 0 ? "M" : "L"}} ${{point[0]}} ${{point[1]}}`).join(" ");
    }}

    function createSvg(tag, attrs) {{
      const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
      for (const [key, value] of Object.entries(attrs)) {{
        if (value !== "") element.setAttribute(key, value);
      }}
      return element;
    }}

    function fitView() {{
      viewBox = {{
        x: -pad,
        y: -pad,
        w: graphBounds.width + pad * 2,
        h: graphBounds.height + pad * 2
      }};
      applyViewBox();
    }}

    function applyViewBox() {{
      svg.setAttribute("viewBox", `${{viewBox.x}} ${{viewBox.y}} ${{viewBox.w}} ${{viewBox.h}}`);
    }}

    function screenToSvg(clientX, clientY) {{
      const rect = svg.getBoundingClientRect();
      return {{
        x: viewBox.x + (clientX - rect.left) / rect.width * viewBox.w,
        y: viewBox.y + (clientY - rect.top) / rect.height * viewBox.h
      }};
    }}

    function zoomAtCenter(factor) {{
      zoomAt(viewBox.x + viewBox.w / 2, viewBox.y + viewBox.h / 2, factor);
    }}

    function zoomAt(cx, cy, factor) {{
      const minWidth = Math.max(graphBounds.width / 80, 6);
      const maxWidth = Math.max(graphBounds.width * 30, viewBox.w);
      const nextW = clamp(viewBox.w * factor, minWidth, maxWidth);
      const nextH = nextW * (viewBox.h / viewBox.w);
      const rx = (cx - viewBox.x) / viewBox.w;
      const ry = (cy - viewBox.y) / viewBox.h;
      viewBox = {{
        x: cx - nextW * rx,
        y: cy - nextH * ry,
        w: nextW,
        h: nextH
      }};
      applyViewBox();
    }}

    function stopDrag(event) {{
      if (!isDragging) return;
      isDragging = false;
      dragStart = null;
      svg.classList.remove("dragging");
      if (event.pointerId !== undefined && svg.hasPointerCapture(event.pointerId)) {{
        svg.releasePointerCapture(event.pointerId);
      }}
    }}

    function openSequenceDialog() {{
      if (typeof sequenceDialog.showModal === "function") {{
        sequenceDialog.showModal();
      }} else {{
        sequenceDialog.setAttribute("open", "");
      }}
      updateSequenceMarkers();
      setTimeout(() => sequenceInput.focus(), 0);
    }}

    function parseNodeSequence(text) {{
      const quoted = Array.from(text.matchAll(/["']([^"']+)["']/g))
        .map(match => match[1].trim())
        .filter(Boolean);
      if (quoted.length > 0) return quoted;
      return text
        .split(/[\\s,\\[\\]();]+/)
        .map(value => value.trim())
        .filter(Boolean);
    }}

    function updateSequenceMarkers() {{
      const ids = parseNodeSequence(sequenceInput.value);
      const seen = new Set();
      const markerItems = [];
      const missing = [];
      const duplicates = [];
      sequenceIndexByNode.clear();

      ids.forEach((id, index) => {{
        if (!nodeById.has(id)) {{
          missing.push(id);
          return;
        }}
        if (seen.has(id)) {{
          duplicates.push(id);
          return;
        }}
        seen.add(id);
        sequenceIndexByNode.set(id, index);
        markerItems.push({{ id, index, node: nodeById.get(id) }});
      }});

      renderSequenceMarkers(markerItems);
      updateSequenceStatus(ids, markerItems, missing, duplicates);
      if (selectedNodeId && nodeById.has(selectedNodeId)) selectNode(selectedNodeId, false);
    }}

    function renderSequenceMarkers(items) {{
      sequenceLayer.replaceChildren();
      const offsets = [
        [9, -9],
        [12, 0],
        [9, 9],
        [0, 12],
        [-9, 9],
        [-12, 0],
        [-9, -9],
        [0, -12]
      ];
      for (const item of items) {{
        const radius = Math.max(6.5, 5 + String(item.index).length * 1.9);
        const offset = offsets[item.index % offsets.length];
        const group = createSvg("g", {{
          class: "sequence-marker",
          transform: `translate(${{item.node.x}} ${{item.node.y}})`
        }});
        group.appendChild(createSvg("line", {{ x1: 0, y1: 0, x2: offset[0] * 0.62, y2: offset[1] * 0.62 }}));
        group.appendChild(createSvg("circle", {{ cx: offset[0], cy: offset[1], r: radius }}));
        const text = createSvg("text", {{ x: offset[0], y: offset[1] }});
        text.textContent = item.index;
        group.appendChild(text);
        sequenceLayer.appendChild(group);
      }}
    }}

    function updateSequenceStatus(ids, markerItems, missing, duplicates) {{
      if (ids.length === 0) {{
        sequenceStatus.textContent = "No sequence labels.";
        sequenceInfo.className = "empty";
        sequenceInfo.textContent = "Paste node ids with Seq to mark ordered labels from 0.";
        return;
      }}

      sequenceStatus.textContent = `Parsed ${{ids.length}}, marked ${{markerItems.length}}, missing ${{missing.length}}.`;
      sequenceInfo.className = "";
      const preview = markerItems
        .slice(0, 18)
        .map(item => `${{item.index}}: ${{escapeHtml(item.id)}}`)
        .join("<br>");
      const hiddenCount = Math.max(0, markerItems.length - 18);
      const missingNote = missing.length
        ? `<div class="note">Missing: ${{escapeHtml(missing.slice(0, 20).join(", "))}}${{missing.length > 20 ? " ..." : ""}}</div>`
        : "";
      const duplicateNote = duplicates.length
        ? `<div class="note">Duplicate ids ignored after first mark: ${{escapeHtml(duplicates.slice(0, 20).join(", "))}}${{duplicates.length > 20 ? " ..." : ""}}</div>`
        : "";
      const previewNote = preview
        ? `<div class="note">${{preview}}${{hiddenCount ? `<br>... and ${{hiddenCount}} more` : ""}}</div>`
        : "";
      sequenceInfo.innerHTML =
        detailsHtml([
          ["Parsed", ids.length],
          ["Marked", markerItems.length],
          ["Missing", missing.length],
          ["Duplicates", duplicates.length]
        ]) + previewNote + missingNote + duplicateNote;
    }}

    function setDrawMode(enabled) {{
      isDrawMode = enabled;
      toggleDrawButton.classList.toggle("active", isDrawMode);
      svg.classList.toggle("draw-mode", isDrawMode);
      if (!isDrawMode && activeStroke) {{
        finishDrawing({{ pointerId: undefined }});
      }}
      updateDrawingInfo();
    }}

    function beginDrawing(event) {{
      event.preventDefault();
      hideTooltip();
      const point = screenToSvg(event.clientX, event.clientY);
      activeStroke = {{
        color: activeDrawingColor,
        points: [point],
        path: createSvg("path", {{
          class: "drawing-stroke",
          stroke: activeDrawingColor,
          d: strokePath([point])
        }})
      }};
      drawingLayer.appendChild(activeStroke.path);
      if (event.pointerId !== undefined) svg.setPointerCapture(event.pointerId);
    }}

    function extendDrawing(event) {{
      event.preventDefault();
      const point = screenToSvg(event.clientX, event.clientY);
      const last = activeStroke.points[activeStroke.points.length - 1];
      if (Math.hypot(point.x - last.x, point.y - last.y) < 0.8) return;
      activeStroke.points.push(point);
      activeStroke.path.setAttribute("d", strokePath(activeStroke.points));
    }}

    function finishDrawing(event) {{
      if (!activeStroke) return;
      if (activeStroke.points.length < 2) {{
        activeStroke.path.remove();
      }} else {{
        drawings.push(activeStroke.path);
      }}
      activeStroke = null;
      if (event.pointerId !== undefined && svg.hasPointerCapture(event.pointerId)) {{
        svg.releasePointerCapture(event.pointerId);
      }}
      updateDrawingInfo();
    }}

    function strokePath(points) {{
      if (points.length === 1) {{
        const point = points[0];
        return `M ${{point.x}} ${{point.y}} L ${{point.x + 0.01}} ${{point.y + 0.01}}`;
      }}
      return points.map((point, index) => `${{index === 0 ? "M" : "L"}} ${{point.x}} ${{point.y}}`).join(" ");
    }}

    function undoDrawing() {{
      if (activeStroke) {{
        activeStroke.path.remove();
        activeStroke = null;
      }} else {{
        const path = drawings.pop();
        if (path) path.remove();
      }}
      updateDrawingInfo();
    }}

    function clearDrawings() {{
      drawingLayer.replaceChildren();
      drawings.length = 0;
      activeStroke = null;
      updateDrawingInfo();
    }}

    function updateDrawingInfo() {{
      drawingInfo.className = "";
      drawingInfo.innerHTML = detailsHtml([
        ["Mode", isDrawMode ? "On" : "Off"],
        ["Color", activeDrawingColor],
        ["Strokes", drawings.length]
      ]);
    }}

    function selectNode(nodeId, center) {{
      selectedNodeId = nodeId;
      selectedEdgeId = null;
      const node = nodeById.get(nodeId);
      document.getElementById("nodeSearch").value = nodeId;
      selection.className = "";
      selection.innerHTML = detailsHtml([
        ["Node", node.id],
        ["Longitude", node.lon.toFixed(8)],
        ["Latitude", node.lat.toFixed(8)],
        ["Sequence", sequenceIndexByNode.has(node.id) ? sequenceIndexByNode.get(node.id) : ""],
        ["In degree", node.inDegree],
        ["Out degree", node.outDegree],
        ["Street count", node.attrs.street_count || ""],
        ["Highway", node.attrs.highway || ""],
        ["Ref", node.attrs.ref || ""]
      ]);
      highlightNode(nodeId);
      if (center) centerOn(node.x, node.y);
    }}

    function selectEdge(edgeId) {{
      selectedNodeId = null;
      selectedEdgeId = edgeId;
      const edge = edgeById.get(edgeId);
      selection.className = "";
      selection.innerHTML = detailsHtml([
        ["Edge", `${{edge.source}} -> ${{edge.target}}`],
        ["Weight", edge.weight],
        ["Name", edge.attrs.name || ""],
        ["Highway", edge.attrs.highway || ""],
        ["Oneway", edge.attrs.oneway || ""],
        ["Max speed", edge.attrs.maxspeed || ""],
        ["OSM id", edge.attrs.osmid || ""]
      ]);
      highlightEdge(edgeId);
    }}

    function clearSelection() {{
      selectedNodeId = null;
      selectedEdgeId = null;
      selection.className = "empty";
      selection.textContent = "Click a node to inspect longitude and latitude.";
      document.getElementById("nodeSearch").value = "";
      restoreHighlight();
    }}

    function detailsHtml(rows) {{
      return `<dl>${{rows.filter(([, value]) => value !== "").map(([key, value]) => `<dt>${{escapeHtml(String(key))}}</dt><dd>${{escapeHtml(String(value))}}</dd>`).join("")}}</dl>`;
    }}

    function centerOn(x, y) {{
      viewBox.x = x - viewBox.w / 2;
      viewBox.y = y - viewBox.h / 2;
      applyViewBox();
    }}

    function highlightNode(nodeId) {{
      clearClasses();
      const ids = incidentEdgeIds.get(nodeId) || [];
      for (const [id, element] of edgeElements) {{
        element.classList.toggle("dim", !ids.includes(id));
        element.classList.toggle("highlight", ids.includes(id));
        element.setAttribute("marker-end", ids.includes(id) ? "url(#arrowHi)" : (payload.graph.directed ? "url(#arrow)" : ""));
      }}
      for (const [id, element] of nodeElements) {{
        element.classList.toggle("dim", id !== nodeId);
        element.classList.toggle("highlight", id === nodeId);
      }}
    }}

    function highlightEdge(edgeId) {{
      clearClasses();
      const edge = edgeById.get(edgeId);
      for (const [id, element] of edgeElements) {{
        element.classList.toggle("dim", id !== edgeId);
        element.classList.toggle("highlight", id === edgeId);
        element.setAttribute("marker-end", id === edgeId ? "url(#arrowHi)" : (payload.graph.directed ? "url(#arrow)" : ""));
      }}
      for (const [id, element] of nodeElements) {{
        const active = id === edge.source || id === edge.target;
        element.classList.toggle("dim", !active);
        element.classList.toggle("highlight", active);
      }}
    }}

    function restoreHighlight() {{
      hideTooltip();
      if (selectedNodeId) {{
        highlightNode(selectedNodeId);
      }} else if (selectedEdgeId) {{
        highlightEdge(selectedEdgeId);
      }} else {{
        clearClasses();
      }}
    }}

    function clearClasses() {{
      for (const element of edgeElements.values()) {{
        element.classList.remove("dim", "highlight");
        element.setAttribute("marker-end", payload.graph.directed ? "url(#arrow)" : "");
      }}
      for (const element of nodeElements.values()) {{
        element.classList.remove("dim", "highlight");
      }}
    }}

    function showTooltip(event, html) {{
      tooltip.innerHTML = html;
      tooltip.style.left = `${{event.clientX}}px`;
      tooltip.style.top = `${{event.clientY}}px`;
      tooltip.classList.add("show");
    }}

    function hideTooltip() {{
      tooltip.classList.remove("show");
    }}

    function nodeTooltip(node) {{
      return `<strong>${{escapeHtml(node.id)}}</strong><br>lon ${{node.lon.toFixed(7)}}<br>lat ${{node.lat.toFixed(7)}}`;
    }}

    function edgeTooltip(edge) {{
      const name = edge.attrs.name ? `<br>${{escapeHtml(edge.attrs.name)}}` : "";
      return `<strong>${{escapeHtml(edge.source)}} -> ${{escapeHtml(edge.target)}}</strong><br>${{escapeHtml(edge.weight)}}${{name}}`;
    }}

    function escapeHtml(value) {{
      return value.replace(/[&<>"']/g, char => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }}[char]));
    }}

    function escapeAttr(value) {{
      return escapeHtml(value).replace(/`/g, "&#96;");
    }}

    function clamp(value, min, max) {{
      return Math.max(min, Math.min(max, value));
    }}
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an interactive HTML viewer for a GraphML road network.")
    parser.add_argument("--graphml", type=Path, default=DEFAULT_GRAPHML, help="Input GraphML path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output HTML path.")
    parser.add_argument("--weight-attr", default="length", help="Edge attribute to show as the edge weight.")
    parser.add_argument("--open", action="store_true", help="Open the generated HTML in the default browser.")
    args = parser.parse_args()

    graphml = args.graphml.resolve()
    output = args.output.resolve()
    nodes, raw_edges, directed = parse_graphml(graphml)
    bounds = project_nodes(nodes)
    edges = build_edges(raw_edges, nodes, bounds, args.weight_attr)
    payload = graph_payload(graphml, nodes, edges, bounds, directed, args.weight_attr)
    write_html(payload, output)

    print(f"Wrote {output}")
    print(f"Nodes: {len(nodes)}, edges: {len(edges)}, weight: {args.weight_attr}")
    if args.open:
        webbrowser.open(output.as_uri())


if __name__ == "__main__":
    main()
