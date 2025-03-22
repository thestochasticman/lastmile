from typing_extensions import Self
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pandas import DataFrame
from pprint import pprint
import networkx as nx
import numpy as np
import math

def sort_dict_by_values(d): return dict(sorted(d.items(), key=lambda item: item[1]))

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
    s.G = nx.MultiDiGraph()
    s.pos = {'Exchange': s.origin}
    s.G.add_node('Exchange', pos=s.origin)

  def dist(s: Self, node1: str, node2: str):
    dist = np.hypot(
      s.pos[node1][0] - s.pos[node2][0],
      s.pos[node1][1] - s.pos[node2][1]
    )
    return dist
  
  def arcdist(s: Self, node1: str, node2: str):
    x1, y1 = s.pos[node1][0], s.pos[node1][1]
    x2, y2 = s.pos[node2][0], s.pos[node2][1]

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

  def add_fiber_from_exchange_to_default_ir(s: Self):
    for node in s.G.nodes:
      if node.startswith('I_R'):
        s.G.add_edge('Exchange', node, length=s.dist('Exchange', node), type='fiber_1')

  def add_fiber_from_exchange_to_extra_ir_in_third_ring(s: Self):
    ir3 = [node for node in s.G.nodes if node.startswith('I_R3')]
    ir3_mid = [node for node in ir3 if node.startswith('I_R3_M')]
    ir3_old = [node for node in ir3 if not node.startswith('I_R3_M')]

    for node in ir3_mid:
      dists = {}
      for old_ir_node in ir3_old: dists[old_ir_node] = s.arcdist(old_ir_node, node)
      selected = next(iter(sort_dict_by_values(dists)))
      s.G.add_edge(selected, node, length=dists[selected], type='fiber_2')
      s.G.add_edge('Exchange', selected, length=s.dist('Exchange', selected), type='fiber_2')
  
  def add_fiber_from_exchange_to_extra_ir_in_fourth_ring(s: Self):
    ir4 = [node for node in s.G.nodes if node.startswith('I_R4')]
    ir4_mid = [node for node in ir4 if node.startswith('I_R4_M')]
    ir4_old = [node for node in ir4 if not node.startswith('I_R4_M')]

    visited = []
    for node in ir4_mid:
      dists = {}
      for old_ir_node in ir4_old: dists[old_ir_node] = s.arcdist(old_ir_node, node)
      selected = next(iter(sort_dict_by_values(dists)))
      if selected in visited:
        s.G.add_edge(selected, node, length=dists[selected], type='fiber_4')
        s.G.add_edge('Exchange', selected, length=s.dist('Exchange', selected), type='fiber_4')
      else:
        s.G.add_edge(selected, node, length=dists[selected], type='fiber_3')
        s.G.add_edge('Exchange', selected, length=s.dist('Exchange', selected), type='fiber_3')
        visited += [selected]

  def connect_new_ir_to_closest_ir(s: Self):
    ir = [node for node in s.G.nodes if node.startswith('I_R')]
    old_ir = [node for node in ir if 'M' not in node]
    new_ir = [node for node in ir if 'M' in node]

    for node in new_ir:
      dists = {}
      ring = node[3]
      for old_ir_node in [n for n in old_ir if n[3] == ring]:
        dists[old_ir_node] = s.arcdist(old_ir_node, node)
      selected = next(iter(sort_dict_by_values(dists)))

      s.G.add_edge(selected, node, length=dists[selected], type='fiber_2')
      s.G.add_edge('Exchange', selected, length=s.dist('Exchange', selected), type='fiber_2')   

  def add_driveway_nodes(s: Self):
    for radius in s.radii_km:
      for cross in range(s.segments_per_ring):
        start_angle = (cross / s.segments_per_ring) * 2 * np.pi
        end_angle = ((cross + 1) / s.segments_per_ring) * 2 * np.pi
        for i in range(1, 5):
          angle_dot = start_angle + (i / 5) * (end_angle - start_angle)
          road_x = radius * np.cos(angle_dot)
          road_y = radius * np.sin(angle_dot)
          driveway_name = f"DRW_{radius}_{cross}_{i}"
          s.G.add_node(driveway_name, pos=(road_x, road_y))
          s.pos[driveway_name] = (road_x, road_y)

  
  def add_copper_from_ir_to_two_drw(s: Self):
    ir = [n for n in s.G.nodes if n.startswith('I_R')]
    drw_nodes = [n for n in s.G.nodes if n.startswith('DRW_')]
    for ir_node in ir:
      dists = {}
      for drw_node in drw_nodes:
        dists[drw_node] = s.dist(ir_node, drw_node)
      
      dist_iter = iter(sort_dict_by_values(dists))
      for i in range(2):
        selected = next(dist_iter)
        s.G.add_edge(ir_node, selected, length=s.arcdist(selected, ir_node), type='copper_1')


  def plot_dotted_circle(s: Self, radius, colour='blue'):
    theta = np.linspace(0, 2 * np.pi, 300)
    x = s.origin[0] + radius * np.cos(theta)
    y = s.origin[1] + radius * np.sin(theta)

    plt.plot(x, y, linestyle=':', color=colour, linewidth=2)  # Dotted line

    angles = [180, 45]
    for angle in angles:

      label_x = s.origin[0] + radius * np.cos(angle)
      label_y = s.origin[1] + radius * np.sin(angle)
      plt.text(label_x, label_y, radius, fontsize=10, ha='left', va='bottom', color='black')

  def plot_rings(s: Self, colour='gray'):
    for radius in s.radii_km: s.plot_dotted_circle(radius, colour=colour)

  def plot_edges(
    s: Self,
    edge_type: str,
    colour: str,
    connection_style: str,
    should_label=False,
    label_pos=0.5
  ):
    edges = [(u, v) for u, v, d in s.G.edges(data=True) if d.get('type') == edge_type]
    nx.draw_networkx_edges(
      s.G,
      s.pos,
      edgelist=edges,edge_color=colour,
      connectionstyle=connection_style,
      arrows=True,
      width=3
    )
    # if should_label:
      # labels = {(u, v): round(d['length'], 2) for u, v, d in s.G.edges(data=True) if 'length' in d and d.get('type') == edge_type}
      # nx.draw_networkx_edge_labels(s.G, s.pos, edge_labels=labels, font_size=7, label_pos=label_pos)

  def plot(s: Self):
    labels = {}
    for node in s.G.nodes:
      labels[node] = node

    plt.figure(figsize=(10, 10))
    plt.gca().set_aspect('equal', adjustable='box')
    s.plot_rings(colour='gray')
    node_colours = []
    for n in c.G.nodes:
      if n == 'Exchange': node_colours.append('black')
      elif n.startswith('I_R4_M'): node_colours.append('purple')
      elif n.startswith('I_R3_M'): node_colours.append('purple')
      elif n.startswith('I_R'): node_colours.append('orange')
      elif n.startswith('DRW'): node_colours.append('red')

    # node_sizes = [200 if n == 'Exchange' else 20 for n in c.G.nodes]
    node_sizes = []
    for n in c.G.nodes:
      if n == 'Exchange': node_sizes.append(200)
      elif n.startswith('I_R'): node_sizes.append(100)
      else: node_sizes.append(20)

    nx.draw_networkx_nodes(c.G, c.pos, node_size=node_sizes, node_color=node_colours)
    s.plot_edges('fiber_1', 'orange', 'arc3,rad=0.3', True, 0.1)
    s.plot_edges('fiber_2', 'violet', 'arc3,rad=-0.3', True, 0.1)
    c.plot_edges('fiber_3', 'violet', 'arc3,rad=0.5', True, 0.1)
    c.plot_edges('fiber_4', 'violet', 'arc3,rad=-0.5', True, 0.1)
    
    plt.savefig('simple_fttn.png')
    plt.show()
  
if __name__ == '__main__':
  c = Circle()
  c.add_intersection_nodes()
  c.add_fiber_from_exchange_to_default_ir()
  c.add_one_extra_intersection_node_at_third_ring()
  c.add_two_extra_intersection_nodes_at_fourth_ring()
  c.add_fiber_from_exchange_to_extra_ir_in_third_ring()
  c.add_fiber_from_exchange_to_extra_ir_in_fourth_ring()
  c.add_driveway_nodes()
  c.add_copper_from_ir_to_two_drw()
  c.plot()
