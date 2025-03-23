from typing_extensions import Self
from dataclasses import dataclass
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.pylab import Axes
import networkx as nx
import numpy as np
import heapq
import math

def sort_dict_by_values(d): return dict(sorted(d.items(), key=lambda item: item[1]))

@dataclass
class Circle:
  origin: tuple[float]=(0, 0)
  radii_km: list[float]=np.array([0.75, 1.5, 2.25, 3.0])
  segments_per_ring: int=8
  size_exchange_node: int = 600
  size_intersection_node: int = 300
  size_driveway_node: int = 20
  cost_fiber: int = 10
  cost_copper: int = 6
  cost_fiber_termination: int = 300
  cost_copper_termination: int = 150

  ir_nodes = property(lambda s: [n for n in s.G.nodes if n.startswith('I_R')])
  drw_nodes = property(lambda s: [n for n in s.G.nodes if n.startswith('DRW')])

  origin_x = property(lambda s: s.origin[0])
  origin_y = property(lambda s: s.origin[1])


  def __post_init__(s: Self):
    s.G = nx.MultiDiGraph()
    s.exchange = ' '.join(['Exchange', str(s.origin)])
    s.pos = {s.exchange: s.origin}
    s.G.add_node(s.exchange, pos=s.origin, colour='black', size=s.size_exchange_node)

  def dist(s: Self, node1: str, node2: str):
    dist = np.hypot(
      s.pos[node1][0] - s.pos[node2][0],
      s.pos[node1][1] - s.pos[node2][1]
    )
    return round(dist, 3)
  
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
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    theta = math.acos(cos_theta) 
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
        s.G.add_node(name, pos=(x,y), colour='orange', size=s.size_intersection_node)
  
  def add_extra_intersections_on_ring(self, ring: int, per_segment: int):
    radius = self.radii_km[ring - 1]
    for seg in range(self.segments_per_ring):
      angle_start = (seg / self.segments_per_ring) * 2 * np.pi
      angle_end = ((seg + 1) / self.segments_per_ring) * 2 * np.pi
      for i in range(1, per_segment + 1):
        frac = i / (per_segment + 1)  # e.g., 1/4, 1/2, 3/4 for per_segment = 3
        mid_angle = angle_start + frac * (angle_end - angle_start)
        x = radius * np.cos(mid_angle) + self.origin_x
        y = radius * np.sin(mid_angle) + self.origin_y
        name = f"I_R{ring}_M{seg}_{i}"
        self.pos[name] = (x, y)
        self.G.add_node(
          name,
          pos=(x, y),
          colour='purple',
          size=self.size_intersection_node
        )

  def add_fiber_from_exchange_to_default_ir(s: Self):
    for node in s.G.nodes:
      if node.startswith('I_R'):
        s.G.add_edge(s.exchange, node, length=s.dist(s.exchange, node), type='fiber_1')

  def add_fiber_from_exchange_to_extra_ir(s: Self, ring: str):
    ir = [node for node in s.G.nodes if node.startswith('I_R' + str(ring))]
    ir_mid = [node for node in ir if node.startswith('I_R' + str(ring) + '_M')]
    ir_old = [node for node in ir if not node.startswith('I_R' + str(ring) + '_M')]

    visited = []
    for node in ir_mid:
      dists = {}
      for old_ir_node in ir_old: dists[old_ir_node] = s.arcdist(old_ir_node, node)
      selected = next(iter(sort_dict_by_values(dists)))
      if selected in visited:
        s.G.add_edge(selected, node, length=dists[selected], type='fiber_4')
        s.G.add_edge(s.exchange, selected, length=s.dist(s.exchange, selected), type='fiber_4')
      else:
        s.G.add_edge(selected, node, length=dists[selected], type='fiber_3')
        s.G.add_edge(s.exchange, selected, length=s.dist(s.exchange, selected), type='fiber_3')
        visited += [selected]


  def connect_new_ir_to_closest_ir(s: Self, ring):
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
      s.G.add_edge(s.exchange, selected, length=s.dist(s.exchange, selected), type='fiber_2')   

  def add_driveway_nodes(s: Self):
    for radius in s.radii_km:
      for cross in range(s.segments_per_ring):
        start_angle = (cross / s.segments_per_ring) * 2 * np.pi
        end_angle = ((cross + 1) / s.segments_per_ring) * 2 * np.pi
        for i in range(1, 5):
          angle_dot = start_angle + (i / 5) * (end_angle - start_angle)
          road_x = radius * np.cos(angle_dot)
          road_y = radius * np.sin(angle_dot)
          road_x = road_x + s.origin_x
          road_y = road_y + s.origin_y
          driveway_name = f"DRW_{radius}_{cross}_{i}"
          s.G.add_node(driveway_name, pos=(road_x, road_y), colour='red', size=s.size_driveway_node)
          s.pos[driveway_name] = (road_x, road_y)

  def add_copper_to_each_drw_from_closest_ir(s: Self):
    ir_nodes = [n for n in s.G.nodes if n.startswith('I_R')]
    drw_nodes = [n for n in s.G.nodes if n.startswith('DRW_')]

    for drw_node in drw_nodes:
      dists = {}
      _dists = []
      for ir_node in ir_nodes:
        dists[ir_node] = s.dist(ir_node, drw_node)
        # print(s.pos[drw_node], s.pos[ir_node], dists[ir_node])

      sorted_dists = sort_dict_by_values(dists)
      selected = next(iter(sorted_dists))
      s.G.add_edge(selected, drw_node, length=s.arcdist(selected, drw_node), type='copper_1')
  
  def check_if_all_driveways_connected(s: Self):
    return all([nx.has_path(s.G, s.exchange, n) for n in s.drw_nodes])
  
  def compute_total_fiber_nodes(s: Self):
    return len([node for node in s.G.nodes if node.startswith('I_R')])
  
  def compute_total_fiber_in_km(s: Self):
    return sum([d.get('length') for u, v, d in s.G.edges(data=True) if d.get('type').startswith('fiber')])
  
  def compute_total_copper_in_km(s: Self):
    return sum([d.get('length') for u, v, d in s.G.edges(data=True) if d.get('type').startswith('copper')])
  
  def compute_total_fiber_cost(s: Self):
    return s.compute_total_fiber_in_km() * s.cost_fiber * 1000
  
  def compute_total_copper_cost(s: Self):
    return s.compute_total_copper_in_km() * s.cost_copper * 1000
  
  def compute_cost_fiber_termination(s: Self):
    print(len(s.ir_nodes))
    return len(s.ir_nodes) * s.cost_fiber_termination
  
  def compute_cost_copper_termination(s: Self):
    print(len(s.drw_nodes))
    return len(s.drw_nodes) * s.cost_copper_termination

  def estimate_speed(s: Self, x: float):
    x = x * 1000
    base_speed = 300.0   # Mbps at 0m
    k = math.log(6) / 600.0  # ensures speed(600m) ≈ 50 Mbps
    return base_speed * math.exp(-k * x)
  
  def estimate_speed_per_drw(s: Self):
    copper_edge_info = [(v, d.get('length')) for u, v, d in s.G.edges(data=True) if d.get('type').startswith('copper')]
    speeds = {}
    for drw, dist in copper_edge_info:
      speeds[drw] = (s.estimate_speed(dist))
    s.speeds = speeds
    return speeds
  
  node_colours = property(lambda s: [colour for node, colour in s.G.nodes(data='colour')])
  node_sizes = property(lambda s: [size for node, size in s.G.nodes(data='size')])
  legend_elements = property(lambda s: [
      Line2D([0], [0], color='lightgray', lw=2, label='Roads'),
      Line2D([0], [0], color='orange', lw=2, label='Fiber Cable (Intersection)'),
      Line2D([0], [0], color='violet', lw=2, label='Fiber Cable (Non Intersection)'),
      Line2D([0], [0], color='brown', lw=2, label='Copper Cable'),
      Patch(facecolor='Black', edgecolor='black', label=s.exchange),
      Patch(facecolor='red', edgecolor='black', label='Driveway (DRW)'),
      Patch(facecolor='orange', edgecolor='black', label='FTTN Node (Intersection)'),
      Patch(facecolor='Purple', edgecolor='black', label='FTTN Node (Not Intersection)')
    ]
  )

  def plot_dotted_circle(s: Self, ax: Axes, radius, colour='blue'):
    theta = np.linspace(0, 2 * np.pi, 300)
    x = s.origin[0] + radius * np.cos(theta)
    y = s.origin[1] + radius * np.sin(theta)

    ax.plot(x, y, linestyle=':', color=colour, linewidth=2)  # Dotted line

    angles = [np.pi / 2]
    for angle in angles:
      label_x = s.origin[0] + radius * np.cos(angle)
      label_y = s.origin[1] + radius * np.sin(angle)
      plt.text(label_x, label_y, radius, fontsize=20, ha='left', va='bottom', color='black')

  def plot_rings(s: Self, ax: Axes, colour='gray', ):
    for radius in s.radii_km: s.plot_dotted_circle(ax, radius, colour=colour)

  def plot_edges(
    s: Self,
    ax: Axes,
    edge_type: str,
    colour: str,
    connection_style: str,
    width: float=3,
  ):
    edges = [(u, v) for u, v, d in s.G.edges(data=True) if d.get('type') == edge_type]
    nx.draw_networkx_edges(
      s.G,
      s.pos,
      ax=ax,
      edgelist=edges,edge_color=colour,
      connectionstyle=connection_style,
      arrows=True,
      width=width,
      arrowstyle='-|>'
    )

  def plot_radial_segments(self, ax: Axes, color='lightgray', linewidth=1.0):
    for i in range(self.segments_per_ring):
      angle = (i / self.segments_per_ring) * 2 * np.pi
      x0, y0 = self.origin
      x1 = x0 + self.radii_km[-1] * np.cos(angle)
      y1 = y0 + self.radii_km[-1] * np.sin(angle)
      ax.plot([x0, x1], [y0, y1], color=color, linewidth=linewidth)

  def plot(s: Self):
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_aspect('equal', adjustable='box')
    nx.draw_networkx_nodes(s.G, s.pos, s.G.nodes, s.node_sizes, s.node_colours, ax=ax)
    s.plot_radial_segments(ax)
    s.plot_rings(ax)
    
    s.plot_edges(ax, 'fiber_1', 'orange', 'arc3,rad=0.3', width=2)
    s.plot_edges(ax, 'fiber_2', 'violet', 'arc3,rad=-0.3', width=2)
    c.plot_edges(ax, 'fiber_3', 'violet', 'arc3,rad=0.5', width=2)
    c.plot_edges(ax, 'fiber_4', 'violet', 'arc3,rad=-0.5', width=2)
    c.plot_edges(ax, 'copper_1', 'brown', 'arc3,rad=1.0', width=2)
    ax.legend(handles=s.legend_elements, loc='lower right', fontsize='16')
    ax.set_title('FTTN Peer 2 Peer + Copper Cable', fontsize=20)
    fig.text(0.5, 0.02, 'Fig 1.6', ha='right', fontsize=30, color='gray')
    fig.tight_layout()
    fig.savefig('simple_fttn.png')
    plt.show()

  def plot_speeds_histogram(s: Self):
    fig, ax = plt.subplots(figsize=(16, 16))
    # ax.set_aspect('equal', adjustable='box')
    speeds = list(s.speeds.values())
    # ax.hist(x=speeds, bins=5, histtype='stepfilled', facecolor='blue', linewidth=2, edgecolor='black')
    counts, bins, patches = ax.hist(speeds, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    for bin_edge in bins:
      ax.axvline(bin_edge, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(x=50, color='red', linestyle=':', linewidth=2)
    plt.show()
  
if __name__ == '__main__':
  c = Circle()
  c.add_intersection_nodes()
  c.add_fiber_from_exchange_to_default_ir()
  c.add_extra_intersections_on_ring(ring=2, per_segment=1)
  c.add_extra_intersections_on_ring(ring=3, per_segment=1)
  c.add_extra_intersections_on_ring(ring=4, per_segment=2)
  c.add_fiber_from_exchange_to_extra_ir(ring=2)
  c.add_fiber_from_exchange_to_extra_ir(ring=3)
  c.add_fiber_from_exchange_to_extra_ir(ring=4)
  c.add_driveway_nodes()
  c.add_copper_to_each_drw_from_closest_ir()
  if c.check_if_all_driveways_connected():
    c.estimate_speed_per_drw()
    print('Congratulations !!!!', 'All driveways have connection to exchange')
    print('total fiber in km: ', c.compute_total_fiber_in_km())
    print('total copper in km: ', c.compute_total_copper_in_km())
    avg = sum(list(c.speeds.values())) / len(c.speeds)
    print('average download:', avg)
    print('total fiber cost', c.compute_total_fiber_cost())
    print('total copper cost', c.compute_total_copper_cost())
    print('fiber termination cost', c.compute_cost_fiber_termination())
    print('copper termination cost', c.compute_cost_copper_termination())

    print(
      'total cost: ',
      sum(
        [
          c.compute_total_fiber_cost(),
          c.compute_total_copper_cost(),
          c.compute_cost_fiber_termination(),
          c.compute_cost_copper_termination()
        ]
      )
    )
  c.plot_speeds_histogram()
  
  # c.plot_speeds_histogram()
  


  # c.estimate_speed_per_drw()

  # for node in c.speeds:
  #   if c.speeds[node] < 50:
  #     print(node, c.speeds[node])

  # plt.figure(figsize=(16, 16))
  # plt.hist(c.speeds.values())
  # plt.show()