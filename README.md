# Monet_Paintings_Lodia
Assignment For Generative models: I’m Something of a Painter Myself

WANDB: https://wandb.ai/llodi22-free-university-of-tbilisi-/Monet_Paintings_Lodia/table?nw=nwuserllodi22

მოდი ჯერ ვისაუბროთ რაზეა საერთოდ ქომფეთიშენი:
მიზანია ჩვეულებრივი ფოტოაპარატით გადაღებული სურათები, მონეს ნახატების სტილში დავაგენერიროთ.
photo_jgp/ ფოტოები
monet_jpg/ მონეს ნახატები

საქმე იმაშია რომ ჩვენ არ გვაქვს მონეს ნახატების შესაბიმისი სურათები, ამიტომ supervised learning არ გამოგვივა.

რატომ ვიყენებთ cycleGAN-ს: 
ეს მოდელი აგვარებს იმ პრობლემას რო სურათების წყვილები არგვაქვს, ორმაგი მიმართულებით სწავლებით!
G_P2M ფოტო -> მონე
G_M2P მონე -> ფოტო (ეს დამხმარეა უფრო, ტრენინგის სტაბილურობისთვის)

გამოვიყენებთ 2 დისკრიმინატორს: D_M(აფასებს მიღებულ მონეს ნახატს) და  D_P(აფასებს მიღებულ ფოტოს). 

CycleGAN იყენებს cycle-consistency-ის:
1. ავიღოთ რეალური სურათი P.
2. გადავიყვანოთ მონეში: P → G_P2M(P) = fake_M
3. გადმოვიყვანოთ უკან ფოტოში: fake_M → G_M2P(fake_M) = rec_P
4. ვეცადოთ რომ rec_P და P გავდნენ ერთმანეთს.

ლოგიკა ის დევს, რომ ამ მეთოდით მოდელი უფრო ბევრს ისწავლის როგორც სურათზე ასევე მონეზე.

### პირველი ექსპერიმენტი: U-Net Generator vs ResNet Generator.

1. გენერატორის როლი:

CycleGAN-ში გენერატორს გადაყავს სურათი ერთი დომენიდან მეორეში.
ინახავს სურათის სტრუქტურას და შინაარსს.
სტილის, ტექსტურისა და ფერის განაწილების შეცვლა.
ციკლის თანმიმდევრულობა, ანუ ორიგინალურ გამოსახულებაზე ინვერსიული შესაბამისობის პოვნა.

2. ResNet Generator

აქ ვიყენებ იმ არქიტექტურას რაც CycleGan-ის დოკუმენტაციაში გვხვდება.
Initial convolution with large receptive field
Downsampling using strided convolutions
Multiple residual blocks
Upsampling back to original resolution
Final Tanh output layer

The core building block is the residual block:

y=x+F(x), where F(x) represents a small convolutional transformation.


3. U-Net Generator
The U-Net generator follows an encoder–decoder structure with long skip connections between corresponding layers:

Encoder progressively downsamples the image

Bottleneck captures global representation

Decoder upsamples back to full resolution

Skip connections concatenate encoder features to decoder layers

Unlike ResNet, U-Net connects early low-level features directly to late reconstruction layers.


## next_Monet_Lodia_exp1_v2_full.ipynb (ეს არის დასრულებული ფაილი, კიდევ რამდენიმე ფაილია რომლებიც ბოლომდე ვერ გავიდნენ GPU-ს გამო, მაგრამ ჩექპოინტებით ვაგრძელებდი)
This code is:
CycleGAN with ResNet generator + PatchGAN discriminator + LSGAN loss at 256×256.


# ჯერ ვისაუბროთ დატაზე და preprocessing-ზე:

დატა:დავწერე MonetPhotoDatase, რომ ყოველ სემპლზე მომცეს ერთი ფოტო და ერთი ნახატი.
ყოველ ტრენინგის ნაბიჯზე გვექნება ნამდვილი ფოტო real_P და მონეს ნახატი real_M. ესენი წყვილი არ არი, ფრიად დამოუკიდებლები არიან.
preprocessing: ვიყენებთ - Resize(256,256), RandomHorizontalFlip(0.5) (simple augmentation),ToTensor()Normalize(mean=0.5, std=0.5) per RGB channel
This normalization maps pixel range [0,1] to [-1,1], which matches the generator’s final Tanh() output.

batch_sizeს ვიყენებ ერთს რადგან ოფიციალურ ფეიფერში ეგრე ეწერა. ამცირებს gpuს გამოყენებას 256x256ზე.

# არ გამომიყენებია არც ერთი შეზღუდული tool.
დავწერე: ResNetGenerator(nn.Module), PatchDiscriminator(nn.Module), training loop (manual), loss formulas (explicit).

# ResNet-based Generator:
ngf = 32 (lighter/faster than the common 64)
n_blocks = 9 (recommended for 256×256 in CycleGAN)
n_downsampling = 2

ჩვენი გენერატორი არის encoder–transformer–decoder.
1. Initial conv (7×7)
  Reflection padding to avoid edge artifacts
  Conv → InstanceNorm → ReLU

2.Downsampling ×2
  Each downsampling halves resolution and increases channels:
  256×256 → 128×128 → 64×64
  channels: 32 → 64 → 128
  
3.Residual blocks (9 blocks) at 64×64
  Each block:
    does Conv → INorm → ReLU → Conv → INorm
    then adds skip connection: x + block(x)
    These blocks keep spatial size but learn style/content transformations.

4.Upsampling ×2 (ConvTranspose)
    64×64 → 128×128 → 256×256
    channels go back down: 128 → 64 → 32

5.Output conv (7×7) + Tanh
    output has 3 channels (RGB)
    output range is [-1, 1]

# Discriminator: PatchGAN(70x70)
ეს დისკრიმინატორი არ გვიბრუნებს პასუხად უბრალოდ real/fake-ს. გვიბრუნებს score-ების გრიდს. სხვანაირად რომ ავხსნა ის არ მპასუხობს ეს არის თუ არა მონეს ნახატი, ის მეუბნება კონკრეტული ნაწილები სურათების არის თუ არა მონეს სტილის.
არქიტექტურა: 
We use 4×4 convolutions:
    First layer: Conv(3→32), stride 2, LeakyReLU
    Next layers: Conv with InstanceNorm + LeakyReLU
    Final layer outputs 1-channel map of realism scores

გვიბრუნებს ეგრედ წოდებულ patch map-ს. მაღალი შედეგია სადაც გავს, დაბალია სადაც არ გავს.

# Loss Funcs

ვიყენებ სამნაირ loss-ს.
1. Adversarial loss(LSGAN)
   criterion_GAN = nn.MSELoss()

2. Cycle-consistency loss
   "აკონტროლებს გადასვლებს":
     P → M → P should reconstruct original P
     M → P → M should reconstruct original M
     criterion_cycle = nn.L1Loss()

3. Identity loss
   ვცდილობთ არ დავუშვათ არასიჭირო ფერების ცვლა. 
   ჩვენ რომ მონეს სურათი გავუშვათ მონე->სურათი გენერატორში, ზუსტად იგივე უნდა დაგვიბრუნოს
     G_P2M(M) ≈ M, G_M2P(P) ≈ P
     criterion_identity = nn.L1Loss()


# ტრენინგი

  გენერატორის ტრენინგი
  1. identity loss
  2. adversarial loss for Photo→Monet
  3. adversarial loss for Monet→Photo
  4. cycle losses for both cycles
    Then sum them and backprop once for both generators. his is why our generator loss is called G_total.

  მონეს დისკრიმინატორის ტრენინგი:
  loss_real: D_M(real_M) → 1
  loss_fake: D_M(fake_M.detach()) → 0
  Then average.

  სურათების დისკრიმინატორის ტრენინგი:
  D_P(real_P) → 1
  D_P(fake_P.detach()) → 0
  Then average.

# Run Results:
  D_M = 0.01413
  D_P = 0.25527
  G_M2P_adv = 0.53436
  G_P2M_adv = 1.17031
  G_total = 5.50306
  cycle_loss = 2.58305
  idt_M = 0.45792
  idt_P = 0.75742

  შედეგებიდან ვხედავთ, რომ D_M = 0.014, რაც ძალიან დაბალი და ცუდი შედეგია. ნიშნავს, რომ დისკრიმანატორი მარტივად არჩევს სწორ და ყალბ ნახატებს. გენერატორი მარცხდება. D_P = 0.255 ნიშნავს რომ მონე->ფოტო უფრო რეალისტური გამოვიდა.
  G_M2P_adv = 0.53436
  G_P2M_adv = 1.17031
  ეს ნაწილი ამყარებს ზემოთ ნათქვამს. D_M ძლიერია.
  cycle_loss = 2.58305 ესეც ძალიან მაღალია და ნიშნავს რომ სურათების ციკლი განსხვავდება საწყსისგან.
  idt_M = 0.45792
  idt_P = 0.75742
  ესენი ნიშნავს რომ გენერატორი იმაზე უფრო ცვლის ნახატებს ვიდრე საჭიროა.


## next_Monet_Lodia_U_Net_exp1.ipynb

FID: 93.3667451163031
MiFID: 93.3667451163031

# შევცვალეთ ResNet -> U-Net
ამ ცვლილების მიუხედავად, იდენტური დარჩა:
1. დისკრიმინატორი
2. LSGAN, cycle-consistency, identity
3. დატა და preprocessing
4. 256x256

მიზანია გავარკვიოთ როგორ შეიცვლება Cycle-consistency, Training stability...

U-Net architectures are known to be effective when low-level spatial information must be preserved, which is particularly relevant for artistic style transfer.

# U-Net Generator Architecture
  The U-Net generator follows an encoder–decoder structure with skip connections, where feature maps from the downsampling path are concatenated with corresponding upsampling layers.

1. Downsampling path:
    Series of stride-2 convolutions
    Gradually increases channel depth
    Captures global context and semantics

2. Upsampling path:
     Transposed convolutions
     Gradually restores spatial resolution

3. Skip connections:
    Direct concatenation of encoder features into decoder
    Preserve fine spatial details (edges, textures)

4. Final activation:
     Tanh, producing outputs in [−1,1]

Unlike the ResNet generator, U-Net does not rely on residual blocks for information flow; instead, it explicitly reuses early-layer features, which has a strong impact on visual sharpness.

| Aspect             | ResNet Generator     | U-Net Generator             |
| ------------------ | -------------------- | --------------------------- |
| Core idea          | Residual learning    | Encoder–decoder + skips     |
| Feature reuse      | Implicit (residuals) | Explicit (skip connections) |
| Spatial detail     | Moderately preserved | Strongly preserved          |
| Texture sharpness  | Softer               | Sharper                     |
| Memory usage       | Lower                | Higher                      |
| Training stability | Very stable          | Slightly more oscillatory   |

# Run Results

For U-Net:

| Metric       | Value     |        
| ------------ | --------- |
| G_total    | **2.72**  |
| G_P2M_adv  | 1.12      |
| G_M2P_adv  | 0.67      |
| cycle_loss | **0.75**  |
| idt_M      | **0.07**  |
| idt_P      | **0.11**  |
| D_M        | 0.017     |
| D_P       | **0.036** |

For ResNet:

D_M = 0.01413
D_P = 0.25527
G_M2P_adv = 0.53436
G_P2M_adv = 1.17031
G_total = 5.50306
cycle_loss = 2.58305
idt_M = 0.45792
idt_P = 0.75742

Cycle loss is significantly lower than in the ResNet experiment
→ U-Net reconstructs images more faithfully.

Identity losses are much smaller
→ The generator minimally alters images already in the target domain.

Discriminator losses are low but balanced
→ No discriminator collapse observed.

Overall generator loss is lower than ResNet
→ Training objective is easier for U-Net due to skip connections.


13.1 Photo → Monet Translation

Observed characteristics:

Strong Monet-style color palettes

Clear brush-like textures

Better preservation of edges and object boundaries

Less excessive blurring compared to ResNet

U-Net’s skip connections help retain local structure, which results in sharper stylization.

13.2 Cycle Consistency (P → M → P, M → P → M)

Cycle reconstructions show:

High structural fidelity

Minimal geometric distortion

Very small color drift

This aligns with the low cycle loss (0.75) and confirms that U-Net is particularly effective at reconstruction-based constraints.

13.3 Monet → Photo Translation

Generated photos are sharper than ResNet outputs

Some painterly artifacts remain (expected)

Slightly reduced realism compared to ResNet in certain scenes

This highlights a trade-off:

U-Net prioritizes structure

ResNet slightly prioritizes global realism


---------------------------------------------------------------------------------------
16.1 Summary of Experimental Results

The experiments demonstrated that both architectures successfully learned meaningful mappings between the photo and Monet domains without paired supervision.

The ResNet-based CycleGAN produced visually smooth and globally consistent Monet-style images, capturing color palettes and painterly textures effectively.

The U-Net-based CycleGAN achieved stronger cycle consistency and identity preservation, producing sharper images with clearer structural details.

Quantitatively, the U-Net model achieved:

Lower cycle-consistency loss

Lower identity loss

Lower overall generator loss

While the ResNet model showed:

Slightly better global stylization

More stable adversarial dynamics

These results indicate that architecture choice directly affects the trade-off between stylistic abstraction and structural fidelity.

16.2 What We Learned About CycleGAN

Through this project, several key insights about CycleGAN were observed:

Cycle-consistency is essential
Without the cycle loss, the generators would easily collapse to arbitrary mappings. The low cycle loss values in both experiments confirm that the bidirectional constraint is doing meaningful work.

Identity loss improves color stability
Identity loss helped prevent unnecessary color shifts when an image was already in the target domain, especially noticeable in the U-Net experiment.

Unpaired training is feasible but fragile
Training without paired data works, but requires careful balancing of losses and learning rates to avoid mode collapse or texture artifacts.

Discriminator loss alone is not a quality metric
Very low discriminator loss (e.g., D_M ≈ 0.01) does not necessarily imply better images; visual inspection remains critical.

16.3 Architectural Insights: ResNet vs U-Net

This experiment highlighted how architectural inductive biases shape the learned mapping:

ResNet generators rely on residual learning to modify features gradually, leading to smoother and more painterly outputs.

U-Net generators reuse low-level features via skip connections, which preserves edges and spatial structure but can reduce stylistic abstraction.

In other words:

ResNet favors artistic transformation

U-Net favors structural reconstruction

Neither architecture is universally better; the “best” choice depends on whether the task prioritizes style realism or content fidelity.

16.4 Practical Takeaways

From an engineering perspective, the project provided practical lessons:

Batch size = 1 is critical for CycleGAN stability.

Training time scales heavily with image resolution.

Checkpointing is essential due to long training times and unstable runtimes.

Visual monitoring is necessary; losses alone are insufficient.

16.5 Final Remarks

Overall, this project demonstrates that CycleGAN is a powerful framework for unpaired image translation, capable of learning complex artistic transformations from limited supervision. The comparative study between ResNet and U-Net generators shows that model architecture plays a crucial role in balancing realism, structure, and artistic style.

------------------------------------------------------------------------------------------------------------











LSGAN vs Hinge Loss

FID: 89.65889200145094
MiFID: 103.301895

In this experiment, we compare two different adversarial loss formulations used to train Generative Adversarial Networks (GANs): Least Squares GAN (LSGAN) and Hinge Adversarial Loss.
Both losses define how the generator and discriminator compete during training, but they differ in how strongly and in what regime gradients are applied.

1. Background: Adversarial Loss in GANs

A GAN consists of two networks:

Generator (G): tries to generate fake images that look real

Discriminator (D): tries to distinguish real images from fake ones

The adversarial loss defines:

What it means for the discriminator to be “correct”

How the generator is rewarded or penalized based on discriminator feedback

Different loss functions lead to different training dynamics, stability, and image quality.

2. Least Squares GAN (LSGAN)
Motivation

The original GAN loss (binary cross-entropy) often suffers from:

Vanishing gradients

Unstable training

Saturation when the discriminator becomes too confident

LSGAN was proposed to address these issues by replacing binary classification with a least-squares regression objective.

LSGAN Discriminator Loss

For real images x∼Pdata:
	​
𝐿𝐷𝑟𝑒𝑎𝑙 =(𝐷(𝑥)−1)2 (2 means square)

For fake images G(z): LDfake = (D(G(z)))square

Total discriminator loss = 1/2[LDreal + LDfake]

LSGAN Generator Loss

The generator tries to make fake images look real:


LG=(D(G(z))−1)square

3. Hinge Adversarial Loss
Motivation

Hinge loss comes from margin-based classification, commonly used in SVMs.
Instead of penalizing all mistakes equally, it enforces a margin:

If the discriminator is already confident enough → no loss

Focuses learning on hard examples

This often leads to:

Stronger discriminators

Sharper images

More stable gradients in practice

Hinge Discriminator Loss

For real images:
LDreal=max(0,1−D(x))

For fake images:
LDfake=max(0,1+D(G(z)))

total loss: E[LDreal + Ldfake]

Hinge Generator Loss

The generator simply tries to increase the discriminator score:


LG=−E[D(G(z))]
