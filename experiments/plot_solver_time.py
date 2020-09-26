import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from remat.core.enum_strategy import SolveStrategy

def valid(r):
  return r is not None and r.schedule_aux_data is not None

if __name__ == '__main__':
  result_dict_file = 'data/budget_sweep/p32xlarge_MobileNet_512_None/export_result_dict.pickle'
  result_dict = pickle.load(open(result_dict_file, 'rb'))
  result_dict = {key.name: vals for key, vals in result_dict.items()}

  checkmate_xs, checkmate_ys = \
    zip(*sorted([(r.solver_budget / 1e9, r.solve_time_s) for r in result_dict['OPTIMAL_ILP_GC'] if valid(r)]))
  simrd_xs, simrd_ys = \
    zip(*sorted([(r.solver_budget / 1e9, r.solve_time_s) for r in result_dict['SIMRD'] if valid(r)]))

  sns.set(); sns.set_style('white')
  plt.axhline(y=3600, linestyle='dashed', color='black', label='ILP timeout (1 hour)')
  color, marker, markersize = SolveStrategy.get_plot_params(SolveStrategy.OPTIMAL_ILP_GC)
  plt.plot(checkmate_xs, checkmate_ys, label='', color=color)
  plt.scatter(checkmate_xs,checkmate_ys, label='', s=markersize**2, c=color, marker=marker)
  color, marker, markersize = SolveStrategy.get_plot_params(SolveStrategy.SIMRD)
  plt.plot(simrd_xs, simrd_ys, label='', color=color)
  plt.scatter(simrd_xs, simrd_ys, s=markersize**2, label='', c=color, marker=marker)
  plt.xlabel('Budget (GB)')
  plt.ylabel('Solver Time (s)')
  plt.yscale('log')
  plt.legend(loc='upper right', shadow=False, fancybox=False, framealpha=1.0)
  plt.title('Budget vs. Solver Time (MobileNet)')
  plt.savefig('data/budget_sweep/solve_time.png', bbox_inches='tight', dpi=300)
