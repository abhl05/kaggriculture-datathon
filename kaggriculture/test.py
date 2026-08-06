import json
from kaggle_environments import make
import kagg_control as kcx
plan = kcx.DayPlan("MELON", 8, True, None, 0, False, 100)
env = make("kaggriculture", configuration={"episodeSteps":720,"seed":1})
env.run([lambda o: kcx.act(o,o["player"],plan)[0], "random"])
json.dump(env.toJSON(), open("replay.json","w"))