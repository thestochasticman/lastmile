from typing_extensions import Self
from dataclasses import dataclass
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math

def sort_dict_by_values(d):
  return dict(sorted(d.items(), key=lambda item: item[1]))

def add_driveway_nodes(
  G: nx.Graph,
  pos: dict[str, tuple[float]],
  radii_km: list[float],
  segments_per_ring: int
):
  for radius in radii_km:
    for cross in range(segments_per_ring):
      start_angle = (cross / segments_per_ring) * 2 * np.pi
      end_angle = ((cross + 1) / segments_per_ring) * 2 * np.pi
      for i in range(1, 5):
        angle_dot = start_angle + (i / 5) * (end_angle - start_angle)
        road_x = radius * np.cos(angle_dot)
        road_y = radius * np.sin(angle_dot)
        driveway_name = f"DRW_{radius}_{cross}_{i}"
        G.add_node(driveway_name, pos=(road_x, road_y))
        pos[driveway_name] = (road_x, road_y)

def connect_intersection_nodes_to_2_driveways(
  G: nx.Graph,
  pos: dict[str, tuple[float]],
):
  intersection_nodes = [n for n in G.nodes if n.startswith('I_R')]
  drw_nodes = [n for n in G.nodes if n.startswith('DRW_')]
  for intersection_node in intersection_nodes:
    dists = {}
    x, y = pos[intersection_node]
    for road_node in drw_nodes:
      xr, yr = pos[road_node]
      dist = round(np.hypot(xr - x, yr - y), 3)
      dists[road_node] = dist
    dists = sort_dict_by_values(dists)

    i = 0
    for road_node in dists:
      G.add_edge(intersection_node, road_node, type='copper', length=dists[road_node])
      i = i + 1
      if i == 2:
        break

def connect_driveways(
  G: nx.Graph,
  pos: dict[str, tuple[float]]
):
  drw_nodes = [n for n in G.nodes if n.startswith('DRW')]
  
  for drw in drw_nodes:
    neighbours = nx.neighbors(G, drw)
    if neighbours:
      connections = 1
    else:
      connections = 2
    
    for candidates in drw_nodes:
      dists = {}
      x, y = pos[drw]
      for _drw in G.nodes:
        if drw == _drw: continue
        xr, yr = pos[_drw]
        dist = round(np.hypot(xr - x, yr - y), 3)
        dists[_drw] = dist
    dists = sort_dict_by_values(dists)
    i = 0
    for _drw in dists:
      G.add_edge(drw, _drw, type='road', length=dists[_drw])
      i = i + 1
      if i == connections:
        break
  # for drw in drw_nodes:
  #   dists = {}
  #   x, y = pos[drw]
  #   for _drw in G.nodes:
  #     if drw == _drw: continue
  #     xr, yr = pos[_drw]
  #     dist = round(np.hypot(xr - x, yr - y), 3)
  #     dists[_drw] = dist
    
  #   dists = sort_dict_by_values(dists)
  #   for _drw in dists:
  #     G.add_edge(drw, _drw, type='road', length=dists[_drw])
  #     break

@dataclass
class Circle:
  origin: tuple[float]=(0, 0)
  radii_km: list[float]=np.array([0.75, 1.5, 2.25, 3.0])
  segments_per_ring: int=8

  intersection_nodes = property(lambda s: [n for n in s.G.nodes if n.startswith('I_R')])
  origin_x = property(lambda s: s.origin[0])
  origin_y = property(lambda s: s.origin[1])

  total_fiber_cost = property(
    lambda s: sum(
      d['length']
      for _, _, d in s.G.edges(data=True)
      if d.get('type') == 'fiber'
    )
  )

  def __post_init__(s: Self):
    s.G = nx.Graph()
    s.pos = {'Exchange': s.origin}
    s.G.add_node('Exchange', pos=s.origin)

  def dist(s: Self, node1: str, node2: str):
    dist = np.hypot(
      s.pos[node1][0] - s.pos[node2][0],
      s.pos[node1][1] - s.pos[node2][1]
    )
    return round(dist, 3)
  
  def arcdist(s: Self, node1: str, node2: str):
    x1, y1 = s.pos[node1]
    x2, y2 = s.pos[node2]

    radius = np.hypot(
      x1 - s.origin_x,
      x2 - s.origin_y
    )
    dot = x1 * x2 + y1 * y2
    mag1 = math.sqrt(x1**2 + y1**2)
    mag2 = math.sqrt(x2**2 + y2**2)
    cos_theta = dot / (mag1 * mag2)
    
    # Clamp value to avoid floating point issues
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    
    theta = math.acos(cos_theta)  # in radians
    arc_length = radius * theta
    return arc_length
  
  def add_intersection_nodes(s: Self):
    num_circles = len(s.radii_km)
    for ring in range(1, num_circles + 1):
      for cross in range(s.segments_per_ring):
        angle = (cross / s.segments_per_ring) * 2 * np.pi
        x, y = s.radii_km[ring - 1] * np.cos(angle), s.radii_km[ring - 1] * np.sin(angle)
        x = x + s.origin_x
        y = y + s.origin_y
        name = f'I_R{ring}_C{cross}'
        s.pos[name] = (x, y)
        s.G.add_node(name, pos=(x,y))

  def add_edges_from_exchange_to_first_ring(s: Self):
    origin_x, origin_y = s.origin
    for node in s.G.nodes:
      if node.startswith('I_R1'):
        x, y = s.pos[node]
        dist = round(np.hypot(x - origin_x, y-origin_y), 3)
        s.G.add_edge('Exchange', node, length=dist, type='fiber')

  def add_one_extra_intersection_node_at_third_ring(s: Self):
    radius = s.radii_km[2]
    for i in range(s.segments_per_ring):
      angle_start = (i / s.segments_per_ring) * 2 * np.pi
      angle_end = ((i + 1) / s.segments_per_ring) * 2 * np.pi
      mid_angle = (angle_start + angle_end) / 2
      x = radius * np.cos(mid_angle)
      y = radius * np.sin(mid_angle)
      x = x + s.origin_x
      y = y + s.origin_y
      node_name = f"I_R3_M{i}"
      s.pos[node_name] = (x, y)
      s.G.add_node(node_name, pos=(x, y))

  def add_two_extra_intersection_nodes_at_fourth_ring(s: Self):
    outer_radius = s.radii_km[-1]
    for i in range(s.segments_per_ring):
      angle_start = (i / s.segments_per_ring) * 2 * np.pi
      angle_end = ((i + 1) / s.segments_per_ring) * 2 * np.pi

      # First midpoint at 1/3 of the arc
      angle_1 = angle_start + (angle_end - angle_start) / 3
      x1 = outer_radius * np.cos(angle_1)
      y1 = outer_radius * np.sin(angle_1)
      x1 = x1 + s.origin_x
      y1 = y1 + s.origin_y
      name1 = f"I_R4_M{i}_1"
      s.pos[name1] = (x1, y1)
      s.G.add_node(name1, pos=(x1, y1))

      # Second midpoint at 2/3 of the arc
      angle_2 = angle_start + 2 * (angle_end - angle_start) / 3
      x2 = outer_radius * np.cos(angle_2)
      y2 = outer_radius * np.sin(angle_2)
      x2 = x2 + s.origin_x
      y2 = y2 + s.origin_y
      name2 = f"I_R4_M{i}_2"
      s.pos[name2] = (x2, y2)
      s.G.add_node(name2, pos=(x2, y2))

  def add_fiber_from_lower_ir_to_higher_ir(s: Self):
    num_circles = len(s.radii_km)
    for ring in range(1, num_circles + 1):
      i_ring = [n for n in s.G.nodes if n.startswith('I_R' + str(ring)) and 'M' not in n]
      print(i_ring)
      i_next_ring = [n for n in s.G.nodes if n.startswith('I_R' + str(ring + 1)) and 'M' not in n]
      for node1, node2 in zip(i_ring, i_next_ring):
        s.G.add_edge(node1, node2, length=s.dist(node1, node2), type='fiber')

  
    # add_intersection_nodes(s.G, s.radii_km, s.pos, s.segments)

    # add_roads_from_exchange_to_first_ring(s.G, list(s.pos.keys()), s.radii_km[0])
    # add_midpoint_third_ring(s.G, s.pos, s.segments, s.radii_km[2])
    # add_2_extra_intersection_nodes_4th_ring(s.G, s.pos, s.segments, s.radii_km[-1])
    # add_roads_between_rings(s.G, s.pos, s.segments)
    # add_driveway_nodes(s.G, s.pos, s.radii_km, s.segments)
    # connect_intersection_nodes_to_2_driveways(s.G, s.pos)
    # connect_driveways(s.G,s. pos)
  
   