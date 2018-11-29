import pickle
game_data={"position":"N2 N3",
"pocket":["key","knife"],
"money":160
}
save_file=open("data1/save.dat","wb")
pickle.dump(game_data,save_file)
save_file.close()

import pickle
load_file=open("data1/save.dat","rb")
load_game=pickle.load(load_file)
print(load_game)
load_file.close()

import time
print(time.asctime())

import turtle
t=turtle.Pen()
while True:
    for i in range(0,4):
        t.forward(100)
        t.left(90)
    t.reset()