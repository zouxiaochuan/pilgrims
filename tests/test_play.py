from kaggle_environments import make, Environment

env: Environment
env = make("kore_fleets", debug=True)

starter_agent_path = "tests/starter.py"
env.run([starter_agent_path, starter_agent_path, starter_agent_path, starter_agent_path])
res = env.render(mode="html", width=1000, height=800)
# print(res)