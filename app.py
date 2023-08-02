__all__ = ['is_cat','learn','classify_image','categories','image','label','examples','intf']
import gradio as gr
import fastai.vision.all
from fastai.vision.all import *

def is_cat(x): return x[0].isupper()

learn = load_learner('model.pkl')

categories = ('Black Bear','Grizzly Bear','Teddy Bear')

def classify_image(img):
    pre,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))
 
image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
# examples = []
intf = gr.Interface(fn=classify_image, inputs=image, outputs=label)
intf.launch(share=True)

# def greet(name):
#     return "I love you" + name + " 😍!!"
   
# iface = gr.Interface(fn=greet, inputs="text", outputs="text")
# iface.launch()