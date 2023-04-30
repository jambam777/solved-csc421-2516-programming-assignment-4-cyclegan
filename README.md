Download Link: https://assignmentchef.com/product/solved-csc421-2516-programming-assignment-4-cyclegan
<br>
<strong>Submission: </strong>You must submit three files through MarkUs<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>: a PDF file containing your writeup, titled a4-writeup.pdf, and your code cycle_gan.ipynb. Your writeup must be typeset using L<sup>A</sup>TEX.

The programming assignments are individual work. See the Course Information handout<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a> for detailed policies.

You should attempt all questions for this assignment. Most of them can be answered at least partially even if you were unable to finish earlier questions. If you were unable to run the experiments, please discuss what outcomes you might hypothetically expect from the experiments. If you think your computational results are incorrect, please say so; that may help you get partial credit.

<h1>Introduction</h1>

In this assignment, you’ll get hands-on experience coding and training GANs. This assignment is divided into two parts: in the first part, we will implement a specific type of GAN designed to process images, called a Deep Convolutional GAN (DCGAN). We’ll train the DCGAN to generate emojis from samples of random noise. In the second part, we will implement a more complex GAN architecture called CycleGAN, which was designed for the task of <em>image-to-image translation </em>(described in more detail in Part 2). We’ll train the CycleGAN to convert between Apple-style and Windows-style emojis.

In both parts, you’ll gain experience implementing GANs by writing code for the generator, discriminator, and training loop, for each model.

<h1>Part 1: Deep Convolutional GAN (DCGAN)</h1>

For the first part of this assignment, we will implement a <em>Deep Convolutional GAN (DCGAN)</em>. A DCGAN is simply a GAN that uses a convolutional neural network as the discriminator, and a network composed of <em>transposed convolutions </em>as the generator. To implement the DCGAN, we need to specify three things: 1) the generator, 2) the discriminator, and 3) the training procedure. We will develop each of these three components in the following subsections.

<h2>Implement the Discriminator of the DCGAN [10%]</h2>

The discriminator in this DCGAN is a convolutional neural network that has the following architecture:

Discriminator

<ol>

 <li><strong>Padding: </strong>In each of the convolutional layers shown above, we downsample the spatial dimension of the input volume by a factor of 2. Given that we use kernel size <em>K </em>= 5 and stride <em>S </em>= 2, what should the padding be? Write your answer in your writeup, and show your work (e.g., the formula you used to derive the padding).</li>

 <li><strong>Implementation: </strong>Implement this architecture by filling in the __init__ method of the DCDiscriminator class, shown below. Note that the forward pass of DCDiscriminator is already provided for you.</li>

</ol>

<strong>Note: </strong>The function conv in Helper Modules has an optional argument batch_norm: if batch_norm is False, then conv simply returns a torch.nn.Conv2d layer; if batch_norm is True, then conv returns a network block that consists of a Conv2d layer followed by a torch.nn.BatchNorm2d layer. <strong>Use the conv function in your implementation.</strong>

<h2>Generator [10%]</h2>

Now, we will implement the generator of the DCGAN, which consists of a sequence of transpose convolutional layers that progressively upsample the input noise sample to generate a fake image. The generator has the following architecture:

Generator

<ol>

 <li><strong>Implementation: </strong>Implement this architecture by filling in the __init__ method of the DCGenerator class, shown below. Note that the forward pass of DCGenerator is already provided for you.</li>

</ol>

<strong>Note: </strong>The original DCGAN generator uses deconv function to expand the spatial dimension. <a href="https://distill.pub/2016/deconv-checkerboard/">Odena et al.</a> later found the deconv creates checker board artifacts in the generated samples. In this assignment, we will use upcome that consists of an upsampling layer followed by conv2D to replace the deconv module (analogous to the conv function used for the discriminator above) in your generator implementation.

<h2>Training Loop [15%]</h2>

Next, you will implement the training loop for the DCGAN. A DCGAN is simply a GAN with a specific type of generator and discriminator; thus, we train it in exactly the same way as a standard GAN. The pseudo-code for the training procedure is shown below. The actual implementation is simpler than it may seem from the pseudo-code: this will give you practice in translating math to code.

<strong>Algorithm 1 </strong>GAN Training Loop Pseudocode

<table width="624">

 <tbody>

  <tr>

   <td colspan="2" width="624">1: <strong>procedure </strong>TrainGAN2:                    Draw <em>m </em>training examples {<em>x</em><sup>(<a href="#_ftn3" name="_ftnref3">[3]</a>)</sup><em>,…,x</em><sup>(<em>m</em>)</sup>} from the data distribution <em>p<sub>data</sub></em></td>

  </tr>

  <tr>

   <td width="47">3:</td>

   <td width="577"><strong>Draw </strong><em>m </em><strong>noise samples </strong>{<em>z</em><sup>(<a href="#_ftn4" name="_ftnref4">[4]</a>)</sup><em>,…,z</em><sup>(<em>m</em>)</sup>} <strong>from the noise distribution </strong><em>p<sub>z</sub></em></td>

  </tr>

  <tr>

   <td width="47">4:</td>

   <td width="577"><strong>Generate fake images from the noise: </strong><em>G</em>(<em>z</em><sup>(<em>i</em>)</sup>) <strong>for </strong><em>i </em>∈{1<em>,….m</em>}</td>

  </tr>

  <tr>

   <td width="47">5:</td>

   <td width="577"><strong>Compute the (least-squares) discriminator loss:</strong></td>

  </tr>

  <tr>

   <td width="47">6:</td>

   <td width="577">Update the parameters of the discriminator</td>

  </tr>

  <tr>

   <td width="47">7:</td>

   <td width="577"><strong>Draw </strong><em>m </em><strong>new noise samples </strong>{<em>z</em><sup>(1)</sup><em>,…,z</em><sup>(<em>m</em>)</sup>} <strong>from the noise distribution </strong><em>p<sub>z</sub></em></td>

  </tr>

  <tr>

   <td width="47">8:</td>

   <td width="577"><strong>Generate fake images from the noise: </strong><em>G</em>(<em>z</em><sup>(<em>i</em>)</sup>) <strong>for </strong><em>i </em>∈{1<em>,….m</em>}</td>

  </tr>

  <tr>

   <td width="47">9:</td>

   <td width="577"><strong>Compute the (least-squares) generator loss:</strong></td>

  </tr>

  <tr>

   <td width="47">10:</td>

   <td width="577">Update the parameters of the generator</td>

  </tr>

 </tbody>

</table>

<h1>Part 2: CycleGAN</h1>

Now we are going to implement the CycleGAN architecture.

<h2>Motivation: Image-to-Image Translation</h2>

Say you have a picture of a sunny landscape, and you wonder what it would look like in the rain. Or perhaps you wonder what a painter like Monet or van Gogh would see in it? These questions can be addressed through <em>image-to-image translation </em>wherein an input image is automatically converted into a new image with some desired appearance.

Recently, Generative Adversarial Networks have been successfully applied to image translation, and have sparked a resurgence of interest in the topic. The basic idea behind the GAN-based approaches is to use a conditional GAN to learn a mapping from input to output images. The loss functions of these approaches generally include extra terms (in addition to the standard GAN loss), to express constraints on the types of images that are generated.

A recently-introduced method for image-to-image translation called CycleGAN is particularly interesting because it allows us to use <em>un-paired </em>training data. This means that in order to train it to translate images from domain <em>X </em>to domain <em>Y </em>, we do not have to have exact correspondences between individual images in those domains. For example, in the paper that introduced CycleGANs, the authors are able to translate between images of horses and zebras, even though there are no images of a zebra in exactly the same position as a horse, and with exactly the same background, etc.

Thus, CycleGANs enable learning a mapping from one domain <em>X </em>(say, images of horses) to another domain <em>Y </em>(images of zebras) <em>without </em>having to find perfectly matched training pairs.

To summarize the differences between paired and un-paired data, we have:

<ul>

 <li>Paired training data:</li>

 <li>Un-paired training data:

  <ul>

   <li>Source set: with each <em>x</em><sup>(<em>i</em>) </sup>∈ <em>X</em></li>

   <li>Target set: with each <em>y</em><sup>(<em>j</em>) </sup>∈ <em>Y</em></li>

   <li>For example, <em>X </em>is the set of horse pictures, and <em>Y </em>is the set of zebra pictures, where there are no direct correspondences between images in <em>X </em>and <em>Y</em></li>

  </ul></li>

</ul>

<h2>Emoji CycleGAN</h2>

Now we’ll build a CycleGAN and use it to translate emojis between two different styles, in particular, Windows &#x2194; Apple emojis.

<h2>Generator [20%]</h2>

The generator in the CycleGAN has layers that implement three stages of computation: 1) the first stage <em>encodes </em>the input via a series of convolutional layers that extract the image features; 2) the second stage then <em>transforms </em>the features by passing them through one or more <em>residual blocks</em>; and 3) the third stage <em>decodes </em>the transformed features using a series of transpose convolutional layers, to build an output image of the same size as the input.

The residual block used in the transformation stage consists of a convolutional layer, where the input is added to the output of the convolution. This is done so that the characteristics of the output image (e.g., the shapes of objects) do not differ too much from the input.

Implement the following generator architecture by completing the __init__ method of the

CycleGenerator class.

To do this, you will need to use the conv and upconv functions, as well as the ResnetBlock class, all provided in Helper Modules.

CycleGAN Generator

3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  3

<strong>Note: </strong>There are two generators in the CycleGAN model, <em>G<sub>X</sub></em><sub>→<em>Y </em></sub>and <em>G<sub>Y </sub></em><sub>→<em>X</em></sub>, but their implementations are identical. Thus, in the code, <em>G<sub>X</sub></em><sub>→<em>Y </em></sub>and <em>G<sub>Y </sub></em><sub>→<em>X </em></sub>are simply different instantiations of the same class.

<h2>CycleGAN Training Loop [20%]</h2>

Finally, we will implement the CycleGAN training procedure, which is more involved than the procedure in Part 1.

<strong>Algorithm 2 </strong>CycleGAN Training Loop Pseudocode

1: <strong>procedure </strong>TrainCycleGAN

2:          Draw a minibatch of samples {<em>x</em><sup>(1)</sup><em>,…,x</em><sup>(<em>m</em>)</sup>} from domain <em>X </em>3:        Draw a minibatch of samples {<em>y</em><sup>(1)</sup><em>,…,y</em><sup>(<em>m</em>)</sup>} from domain <em>Y </em>4:      Compute the discriminator loss on real images:

5:             Compute the discriminator loss on fake images:

6:          Update the discriminators 7:     Compute the <em>Y </em>→ <em>X </em>generator loss:

<em>n</em>

(<em>Y </em>→<em>X</em>→<em>Y </em>)

<em>cycle</em>

=1

8:               Compute the <em>X </em>→ <em>Y </em>generator loss:

<em>m</em>

(<em>X</em>→<em>Y </em>→<em>X</em>)

<em>cycle</em>

=1

9:             Update the generators

Similarly to Part 1, this training loop is not as difficult to implement as it may seem. There is a lot of symmetry in the training procedure, because all operations are done for both <em>X </em>→ <em>Y </em>and <em>Y </em>→ <em>X </em>directions. Complete the cyclegan_training_loop function, starting from the following section:

There are 5 bullet points in the code for training the discriminators, and 6 bullet points in total for training the generators. Due to the symmetry between domains, several parts of the code you fill in will be identical except for swapping <em>X </em>and <em>Y </em>; this is normal and expected.

<h2>Cycle Consistency</h2>

The most interesting idea behind CycleGANs (and the one from which they get their name) is the idea of introducing a <em>cycle consistency loss </em>to constrain the model. The idea is that when we translate an image from domain <em>X </em>to domain <em>Y </em>, and then translate the generated image <em>back </em>to domain <em>X</em>, the result should look like the original image that we started with.

The cycle consistency component of the loss is the L1 distance between the input images and their <em>reconstructions </em>obtained by passing through both generators in sequence (i.e., from domain <em>X </em>to <em>Y </em>via the <em>X </em>→ <em>Y </em>generator, and then from domain <em>Y </em>back to <em>X </em>via the <em>Y </em>→ <em>X </em>generator).

The cycle consistency loss for the <em>Y </em>→ <em>X </em>→ <em>Y </em>cycle is expressed as follows:

<em>m</em>

<em>                                                                      ,</em>

=1

where <em>λ<sub>cycle </sub></em>is a scalar hyper-parameter balancing the two loss terms: the cycle consistant loss and the GAN loss. The loss for the <em>X </em>→ <em>Y </em>→ <em>X </em>cycle is analogous.

Implement the cycle consistency loss by filling in the following section in CycleGAN training loop.

Note that there are two such sections, and their implementations are identical except for swapping <em>X </em>and <em>Y </em>. You must implement both of them.

<h2>CycleGAN Experiments [15%]</h2>

<ol>

 <li>Train the CycleGAN to translate Apple emojis to Windows emojis in the Training – CycleGAN section of the notebook. The script will train for 10,000 iterations, and saves generated samples in the samples_cyclegan In each sample, images from the source domain are shown with their translations to the right.</li>

</ol>

Include in your writeup the samples from both generators at either iteration 200 and samples from a later iteration.

<ol start="2">

 <li>Change the random seed and train the CycleGAN again. What are the most noticible difference between the <em>similar </em>quality samples from the different random seeds? Explain why there is such a difference?</li>

 <li>Changing the default lambda_cycle hyperparameters and train the CycleGAN again. Try a couple of different values including <em>without </em>the cycle-consistency loss. (i.e. lambda_cycle = 0)</li>

</ol>

For different values of lambda_cycle, include in your writeup some samples from both generators at either iteration 200 and samples from a later iteration. Do you notice a difference between the results with and without the cycle consistency loss? Write down your observations (positive or negative) in your writeup. Can you explain these results, i.e., why there is or isn’t a difference among the experiments?

<h1>What you need to submit</h1>

<ul>

 <li>Your code file: ipynb.</li>

 <li>A PDF document titled a4-writeup.pdf containing samples generated by your DCGAN and CycleGAN models, and your answers to the written questions.</li>

</ul>

<h2>Further Resources</h2>

For further reading on GANs in general, and CycleGANs in particular, the following links may be useful:

<ol>

 <li><a href="https://distill.pub/2016/deconv-checkerboard/">Deconvolution and Checkerboard Artifacts (Odena et al., 2016)</a></li>

 <li><a href="https://arxiv.org/pdf/1703.10593.pdf">Unpaired image-to-image translation using cycle-consistent adversarial networks (Zhu et al., </a><a href="https://arxiv.org/pdf/1703.10593.pdf">2017)</a></li>

 <li><a href="https://arxiv.org/pdf/1406.2661.pdf">Generative Adversarial Nets (Goodfellow et al., 2014)</a></li>

 <li><a href="http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/">An Introduction to GANs in Tensorflow</a></li>

 <li><a href="https://blog.openai.com/generative-models/">Generative Models Blog Post from OpenAI</a></li>

 <li><a href="https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix">Official PyTorch Implementations of Pix2Pix and CycleGAN</a></li>

</ol>

<a href="#_ftnref1" name="_ftn1">[1]</a> <a href="https://markus.teach.cs.toronto.edu/csc321-2018-01">https://markus.teach.cs.toronto.edu/csc321-2018-01</a>

<a href="#_ftnref2" name="_ftn2">[2]</a> <a href="http://cs.toronto.edu/~rgrosse/courses/csc421_2019/syllabus.pdf">http://cs.toronto.edu/</a><a href="http://cs.toronto.edu/~rgrosse/courses/csc421_2019/syllabus.pdf">~</a><a href="http://cs.toronto.edu/~rgrosse/courses/csc421_2019/syllabus.pdf">rgrosse/courses/csc421_2019/syllabus.pdf</a>

<a href="#_ftnref3" name="_ftn3">[3]</a> . <strong>Implementation: </strong>Fill in the gan_training_loop function in the GAN section of the notebook.

There are 5 numbered bullets in the code to fill in for the discriminator and 3 bullets for the generator. Each of these can be done in a single line of code, although you will not lose marks for using multiple lines.

<strong>Experiment [10%]</strong>

<a href="#_ftnref4" name="_ftn4">[4]</a> . We will train a DCGAN to generate Windows (or Apple) emojis in the Training – GAN section of the notebook. By default, the script runs for 5000 iterations, and should take approximately 10 minutes on Colab. The script saves the output of the generator for a fixed noise sample every 200 iterations throughout training; this allows you to see how the generator improves over time. You can stop the training after obtaining satisfactory image samples. <strong>Include in your write-up one of the samples from early in training (e.g., iteration 200) and one of the samples from later in training, and give the iteration number for those samples. Briefly comment on the quality of the samples, and in what way they improve through training.</strong>