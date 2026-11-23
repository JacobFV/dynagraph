"""G1 env sweep wrapper (same as scripts/robots_env_matrix.py, separate process name so other sweeps' cleanup does
not kill it)."""
import runpy, sys
sys.argv[0] = 'scripts/robots_env_matrix.py'
runpy.run_path('scripts/robots_env_matrix.py', run_name='__main__')
