# Monet_Paintings_Lodia
Assignment For Generative models: I’m Something of a Painter Myself

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

## next_Monet_Lodia_exp1_v2_full.ipynb
This code is:
CycleGAN with ResNet generator + PatchGAN discriminator + LSGAN loss at 256×256.

ესეიგი, ჩემი ექსპერიმენტი იქნება U-Net vs ResNet. გამოვიყენებ CycleGAN-ს და ვეცდები შევადარო შედეგები.

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
   დისკრიმინატორს უნდა რომ თუ რეალურია 1ისკენ წავიდეს, ხოლო თუ ფეიკია 0სკენ.
   გენერატორს უნდა რომ ფეიკი გახადოს რეალური.
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









    


