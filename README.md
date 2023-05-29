
# :pouting_woman: :pouting_man: Animated Face Generation using GAN

## ArtifyGAN
This model has been inspired by CartoonGAN. The GAN
(generative adversarial network) framework consists of two
Convolutional networks. First, is the Generator(G) network
whose goal is to produce output that can fool the
Discriminator(D) network. On the other hand, discriminator
aims to detect whether the image is a synthesized version or belongs to a real dataset. In this project, Generator and
Discriminator was designed to produce the cartoon images for
real dataset by preserving the structure of the real image. In
other words, Generator consisted of three main parts i.e.
encoder which encodes the features of the images, transformer
which transformed the image to match the target manifold and
decoder which takes the transformed image as input and
produces the cartoon version for real image. Discriminator was
a convolution network with a sigmoid layer in the output to
detect whether the image belongs to a real dataset or a fake
dataset and provides adversarial loss for G. Main objective was
to optimize the loss function for the generator.  

In the ArtifyGAN,
residual blocks have been increased from 8 to 10 to capture the
content of the real image more effectively in a lesser number of
epochs. Also, removing the extra convolution block helped to
improve the efficiency of the model by reducing the
computation time. Hence, with this proposed model optimal and efficient results
were achieved in lesser number of epochs i.e. at 79 epochs. For more details refer to the report [here.](https://github.com/iqrabismii/DeepLearningProject/blob/main/DATA_255_Project_Report_Group1.pdf)

![Screenshot 2023-05-28 at 1 56 11 PM](https://github.com/iqrabismii/DeepLearningProject/assets/108056063/0fea5773-7cac-4f24-abf1-ac2485eda931)

Also, the results were compared with other pretrained model such as JojoGAN, StyleGAN, AnimeGAN. From the image above we can see the that ArtifyGAN is able to preserve both content as well as comic style of the image. 

## Implementation
To implement the model, refer to this [notebook.](https://github.com/iqrabismii/DeepLearningProject/blob/main/ArtifyGAN.ipynb)


## To generate Cartoon Images

Install the Python Anaconda Environment and download the zip folder from the repository.Using the command prompt, navigate to the directory of the folder.

### For Mac, run the following command 

1. python3 -m venv myenv
2. source myenv/bin/activate

### For windows, run the following command
1. python -m venv myenv
2. myenv\Scripts\activate

### To install dependencies, run the following:
    pip install -r requirements.txt
   
### To Generate Cartoon image, run the following command:
    python cartoon_image.py <path to real image> 
    For eg, python cartoon_image.py /Users/iqrabismi/Desktop/arizona.jpeg
    
 <br>

------------------------------------------------------------------------------------------------------------------------------------

    
 
    


