from circle import sort_dict_by_values
import matplotlib.pyplot as plt
from circle import Circle
from circle import *
import networkx as nx
import numpy as np



def add_midpoint_third_ring(
  G: nx.Graph,
  pos: dict[str, tuple[float]],
  segments_per_ring: int,
  radius=2.25
):
  for i in range(segments_per_ring):
    angle_start = (i / segments_per_ring) * 2 * np.pi
    angle_end = ((i + 1) / segments_per_ring) * 2 * np.pi
    mid_angle = (angle_start + angle_end) / 2

    x = radius * np.cos(mid_angle)
    y = radius * np.sin(mid_angle)
    node_name = f"I_R3_M{i}"
    pos[node_name] = (x, y)
    G.add_node(node_name, pos=(x, y))
  
   
def add_2_extra_intersection_nodes_4th_ring(
  G: nx.Graph,
  pos: dict[str, tuple[float]],
  segments_per_ring: int,
  radius=3.0
):
  # Add 2 midpoint nodes per segment on the 4th ring (radius = 3.0 km)
  outer_radius = radius
  for i in range(segments_per_ring):
    angle_start = (i / segments_per_ring) * 2 * np.pi
    angle_end = ((i + 1) / segments_per_ring) * 2 * np.pi

    # First midpoint at 1/3 of the arc
    angle_1 = angle_start + (angle_end - angle_start) / 3
    x1 = outer_radius * np.cos(angle_1)
    y1 = outer_radius * np.sin(angle_1)
    name1 = f"I_R4_M{i}_1"
    pos[name1] = (x1, y1)
    G.add_node(name1, pos=(x1, y1))

    # Second midpoint at 2/3 of the arc
    angle_2 = angle_start + 2 * (angle_end - angle_start) / 3
    x2 = outer_radius * np.cos(angle_2)
    y2 = outer_radius * np.sin(angle_2)
    name2 = f"I_R4_M{i}_2"
    pos[name2] = (x2, y2)
    G.add_node(name2, pos=(x2, y2))

def add_roads_between_rings(
  G: nx.Graph,
  pos: dict[str, tuple[float]],
  segments_per_ring: int
):
  # Add edges between rings (radial roads)
  for cross in range(segments_per_ring):
    for ring in range(1, 4):
      n1 = f"I_R{ring}_C{cross}"
      n2 = f"I_R{ring + 1}_C{cross}"
      x1, y1 = pos[n1]
      x2, y2 = pos[n2]
      dist = round(np.hypot(x2 - x1, y2 - y1), 3)
      G.add_edge(n1, n2, length=dist, type='road')

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
      G.add_edge(intersection_node, road_node, type='road', length=dists[road_node])
      i = i + 1
      if i == 2:
        break

def connect_driveways(
  G: nx.Graph,
  pos: dict[str, tuple[float]]
):
  drw_nodes = [n for n in G.nodes if n.startswith('DRW')]
  for drw in drw_nodes:
    dists = {}
    x, y = pos[drw]
    for _drw in drw_nodes:
      if drw == _drw: continue
      xr, yr = pos[_drw]
      dist = round(np.hypot(xr - x, yr - y), 3)
      dists[_drw] = dist

    dists = sort_dict_by_values(dists)
    for _drw in dists:
      G.add_edge(drw, _drw, type='road', length=dists[_drw])
      break

def plot(G: nx.Graph, pos: dict):
  # Nodes
  print(len(G.nodes))
  plt.figure(figsize=(8, 8))
  node_colors = []
  for n in G.nodes:
    if n == 'Exchange': node_colors.append('black')
    elif n.startswith('I_R4_M'): node_colors.append('purple')
    elif n.startswith('I_R3_M'): node_colors.append('purple')
    elif n.startswith('I_R'): node_colors.append('blue')
    elif n.startswith('DRW'): node_colors.append('red')

  node_sizes = [200 if n == 'Exchange' else 20 for n in G.nodes]
  nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)

  fiber_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'fiber']
  fiber_edge_labels = {(u, v): round(d['length'], 1) for u, v, d in G.edges(data=True) if 'length' in d and d.get('type') == 'fiber'}
  nx.draw_networkx_edges(G, pos, edgelist=fiber_edges, edge_color='orange', connectionstyle="arc3,rad=0.5", arrows=True)
  nx.draw_networkx_edge_labels(G, pos, edge_labels=fiber_edge_labels, font_size=7, label_pos=0.2)

  # Separate edges by type
  road_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'road']
  driveway_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'driveway']

  # Separate edges by type
  road_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'road']
  driveway_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'driveway']

  nx.draw_networkx_edges(G, pos, edgelist=road_edges, edge_color='gray')
  nx.draw_networkx_edges(G, pos, edgelist=driveway_edges, edge_color='green', style='dashed')

  # Edge labels (distances only for road edges)
  edge_labels = {(u, v): round(d['length'], 1) for u, v, d in G.edges(data=True) if 'length' in d}
  # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
  plt.axis('equal')
  plt.title("Bungenwood Village Network with Farms and Road Dots", fontsize=20)
  plt.grid(False)
  plt.show()
  plt.savefig('circle.png')
  
if __name__ == '__main__':
  c = Circle()
  c.add_intersection_nodes()
  c.add_edges_from_exchange_to_first_ring()
  c.add_one_extra_intersection_node_at_third_ring()
  c.add_two_extra_intersection_nodes_at_fourth_ring()
  c.add_fiber_from_lower_ir_to_higher_ir()
  # add_midpoint_third_ring(c.G, c.pos, c.segments, c.radii_km[2])
  # add_2_extra_intersection_nodes_4th_ring(c.G, c.pos, c.segments, c.radii_km[-1])
  # add_roads_between_rings(c.G, c.pos, c.segments)
  # add_driveway_nodes(c.G, c.pos, c.radii_km, c.segments)
  # connect_intersection_nodes_to_2_driveways(c.G, c.pos)
  # connect_driveways(c.G, c.pos)

  intersection_nodes = [n for n in c.G.nodes if n.startswith('I_R')]
  G_fibre = nx.subgraph(c.G, intersection_nodes + ['Exchange'])
  # plot(G_fibre, c.pos)
  # print(G_fibre)
  plot(c.G, c.pos)
  # nx.subgraph(
  #   c.G,
    
  # )
