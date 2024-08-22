# **PicoGPT: Character-level Language Model**
<p align="justify">
  Pico-GPT is a miniature language model designed to generate text in the style of Shakespeare, built from scratch using PyTorch.
</p>
<div align="center">
  
  N-Parameters | N-Layers | D-Model | N-Heads | D-Head | Batch-Size | Learning Rate
  :---: |:---: |:---: |:---: |:---: |:---: |:---:
  10M | 6 | 384 | 6 | 64 | 64 | 3 x 10<sup>-4</sup>
</div>

## **Dataset**
The model was trained on the tiny-Shakespeare dataset, which consists of a small portion of Shakespeare's works. 
Despite the limited training data, the model can produce text that reflects the style and structure of Shakespearean English. [Source](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

## **Evaluation Metric**
<p align="center">
  <img width="564" alt="Train & Test Loss" src="https://github.com/user-attachments/assets/3dfdaff0-8e5e-44dc-85f6-b9ee74b1f4b3">
</p>

## **Sample Generation**
<p>
  HARTINGSI:<br />
  O, my lords, spend to be end wild kith!
  So would call-bear that me relations,
  Stars'd not too cousin my minly,
  Sin a stroke rud, her no loves to your will,
  Shall not shall trouble be but read injurtice
  A taen and feely neckobard 'body,'
  That in thy enemy, that the woulds resing nature humber than may
  Read it for the talchop of thy body bed.
  Treads those power! I command, that you bring joys,
  Poor good Marcius, ever matterry we,
  Nor dream none lover, in my well sprigh;
  O-in this reqy my healt clast the York,
  Or on my aufield, and bid make my jesty.
  
  ROMEO:<br />
  Your horse, what I know thought; speak, I salain'd, but let there,
  What should I call it his jollus is knees:
  Sly this joy to my fee, flesh,
  And babeys you'ld here is thusband
  You withal: so I have ever loved. I am so at discont;
  Then do true you know of all Annourit is a quick,
  All I give you sight, and win his wordow.
  Let us quit it be die a marreced boaths.
  
  BUSHORD:<br />
  My dear, that prince the hovers that fair me;
  Thing but the nobles weeping on my feat,
  But I am joyful hated my sons and both made more
  Mores and curstood away to my son, I ommore
  Lack may at in Hecrorous. Presu Elbow that else infection.
  O, were I know the people is drug,
  And therefore there in the father of thine ears and hate.
</p>
