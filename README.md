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

FID: 108.6392649002817
MiFID: 115.9367297786193

# ჯერ ვისაუბროთ დატაზე და preprocessing-ზე:

დატა:დავწერე MonetPhotoDatase, რომ ყოველ სემპლზე მომცეს ერთი ფოტო და ერთი ნახატი.
ყოველ ტრენინგის ნაბიჯზე გვექნება ნამდვილი ფოტო real_P და მონეს ნახატი real_M. ესენი წყვილი არ არი, ფრიად დამოუკიდებლები არიან.
preprocessing: ვიყენებთ - Resize(256,256), RandomHorizontalFlip(0.5) (simple augmentation),ToTensor()Normalize(mean=0.5, std=0.5) per RGB channel
This normalization maps pixel range [0,1] to [-1,1], which matches the generator’s final Tanh() output.

batch_sizeს ვიყენებ ერთს რადგან ოფიციალურ ფეიფერში ეგრე ეწერა. ამცირებს gpuს გამოყენებას 256x256ზე.

# არ გამომიყენებია არც ერთი შეზღუდული tool.
დავწერე: ResNetGenerator(nn.Module), PatchDiscriminator(nn.Module), training loop (manual), loss formulas (explicit).

# ResNet-based Generator:
“Generator იღებს ფოტოს, ჯერ compress-ით იღებს გლობალურ ფიჩერებს, შემდეგ 9 residual block-ით ატარებს სტილურ ცვლილებას, მერე decompress-ით აბრუნებს 256×256-ზე და აძლევს RGB გამოსახულებას.”

ngf = 32 (lighter/faster than the common 64)
n_blocks = 9 (recommended for 256×256 in CycleGAN)
n_downsampling = 2

ჩვენი გენერატორი არის encoder–transformer–decoder.
1. Initial conv (7×7)
  Reflection padding to avoid edge artifacts
  Conv → InstanceNorm → ReLU
	სურათის „საწყისი ფიჩერების“ ამოღება.

2.Downsampling ×2
  Each downsampling halves resolution and increases channels:
  256×256 → 128×128 → 64×64
  channels: 32 → 64 → 128
  ლექტორს ეუბნები: “ვაპატარავებთ რეზოლუციას რომ network-მა უფრო გლობალური სტრუქტურა დაინახოს და იაფი გახდეს გამოთვლა.
  
3.Residual blocks (9 blocks) at 64×64
	იდეა: ბლოკი სწავლობს „პატარა კორექციას“ სურათზე და მერე ამ კორექციას ამატებს ორიგინალს.
	
ReflectionPad2d(1) — სურათის კიდეებს „სარკისებურად“ აფართოებს 1 პიქსელით, რომ კონვოლუციისას კიდეებზე არტეფაქტები ნაკლები იყოს.
	
Conv2d(dim, dim, 3x3) — 3x3 კონვოლუცია, არხების რაოდენობა არ იცვლება (dim → dim).
	
InstanceNorm2d(dim) — ნორმალიზაცია თითო სურათზე/არხზე (CycleGAN-ში სტანდარტია, batch size=1-ზე კარგად მუშაობს).
	
ReLU — არახაზოვანი აქტივაცია.
	
მერე ისევ იგივე: pad → conv → instance norm.

9 ბლოკი ნიშნავს: სტანდარტი 256×256 CycleGAN-ზე.

ეს ნაწილი არ ცვლის ზომას, უბრალოდ ფიჩერებს “სტილზე” გადააწყობს.

აქ ხდება რეალური ‘photo→Monet’ სტილის შეცვლა, რადგან ამ ბლოკებში network სწავლობს რა features უნდა დაამატოს/შეცვალოს.”

4.Upsampling ×2 (ConvTranspose)
    64×64 → 128×128 → 256×256
    channels go back down: 128 → 64 → 32

ვაბრუნებთ original resolution-ს, რომ საბოლოოდ ისევ 256×256 სურათი მივიღოთ.”

5.Output conv (7×7) + Tanh
    output has 3 channels (RGB)
    output range is [-1, 1]
	
ისევ 7x7 — ბოლო smooth mapping RGB-ზე.
Tanh() — output-ს მიჰყავს [-1, 1]-ში, რაც ემთხვევა შენს normalize-ს:
ეს აუცილებელია რადგან GAN training-ში ხშირად output normalized range-ში გვინდა

# Discriminator: PatchGAN(70x70)
“ეს დისკრიმინატორი PatchGAN-ია: downsample-ებით იღებს feature maps-ს, ზრდის არხებს, და ბოლოს აბრუნებს patch-wise რეალურობის map-ს, რომ გენერატორს აიძულოს Monet-ის მსგავსი ტექსტურების სწავლა.”

ეს დისკრიმინატორი არ გვიბრუნებს პასუხად უბრალოდ real/fake-ს. გვიბრუნებს score-ების გრიდს. სხვანაირად რომ ავხსნა ის არ მპასუხობს ეს არის თუ არა მონეს ნახატი, ის მეუბნება კონკრეტული ნაწილები სურათების არის თუ არა მონეს სტილის.
არქიტექტურა: 
We use 4×4 convolutions:
    1)First layer: Conv(3→32), stride 2, LeakyReLU
		4x4 კონვოლუცია stride=2 → ზომას ამცირებს 2-ჯერ (downsample).

ndf=32 არის საწყისი ფიჩერების რაოდენობა (64 სტანდარტია, 32 = უფრო სწრაფი).

LeakyReLU იყენებს GAN-ებში სტაბილურობისთვის (ReLU-ზე უკეთ მუშაობს discriminator-ში).
	
2)Next layers: Conv with InstanceNorm + LeakyReLU
	ყოველ ეტაპზე:

არხები იზრდება: 32 → 64 → 128 (nf_mult = 2^n, max 8-მდე)

stride=2 კიდევ ამცირებს ზომას

InstanceNorm2d იყენებ იმიტომ, რომ batch size=1 გაქვს და ეს კარგად მუშაობს CycleGAN-ში.	
  
3)Final layer outputs 1-channel map of realism scores
stride=1 → უკვე აღარ აკლებს ზომას ბევრად, უბრალოდ უფრო “ზუსტად” აფასებს ადგილობრივ ტექსტურებს.

ბოლო conv აბრუნებს 1 არხს → ეს არის patch-wise score map.
	

გვიბრუნებს ეგრედ წოდებულ patch map-ს. მაღალი შედეგია სადაც გავს, დაბალია სადაც არ გავს.

# Loss Funcs

ვიყენებ სამნაირ loss-ს.
1. Adversarial loss(LSGAN)
   criterion_GAN = nn.MSELoss()
   
   def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)


3. Cycle-consistency loss
   "აკონტროლებს გადასვლებს":
     P → M → P should reconstruct original P
     M → P → M should reconstruct original M
     criterion_cycle = nn.L1Loss()

   def l1_loss(pred, target):
    return torch.mean(torch.abs(pred - target))


5. Identity loss
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

ეს Generator არის Encoder–Decoder:

ქვემოთ (encoder) სურათს “აკუმშავს” → იღებს უფრო მაღალი დონის (high-level) ფიჩერებს

ზემოთ (decoder) ისევ “აშლის” → აბრუნებს 256×256 გამოსავალს
და skip connections-ით (შუალედური “გადახტომები”) ინარჩუნებს დეტალებს.


1. Downsampling path:
    Series of stride-2 convolutions
    Gradually increases channel depth
    Captures global context and semantics

	   UNetDown — “დაუნსემპლინგი” (Encoder-ის ნაწილი)
	nn.Conv2d(in_size, out_size, 4, 2, 1)
	
	
	kernel=4, stride=2, padding=1 → სიგანე/სიმაღლე ნახევრდება (H,W → H/2, W/2)
	
	არხები (channels) იზრდება: in_size -> out_size
	
	InstanceNorm2d (თუ normalize=True) → GAN-ში სტაბილურობა
	
	LeakyReLU → ნეგატიურებში gradient არ “კვდება”
	
	Dropout (თუ ჩართულია) → რეგულარიზაცია
	
	ლექტორს უთხარი: “UNetDown არის encoder-ის ბლოკი: ერთ ნაბიჯში აკეთებს downsample-ს და feature 		  		extraction-	ს.”
   

3. Upsampling path:
     Transposed convolutions
     Gradually restores spatial resolution

   UNetUp — “აპსემპლინგი” + skip-თან შერწყმა (Decoder-ის ნაწილი)
nn.ConvTranspose2d(in_size, out_size, 4, 2, 1)


ConvTranspose2d → ზრდის რეზოლუციას ორჯერ (H,W → 2H, 2W)

InstanceNorm + ReLU → სტაბილური და “სუფთა” feature-ები

მერე:

torch.cat((x, skip_input), 1)


encoder-ის შესაბამისი ფიჩერები კონკატენაციით ებმება decoder-ზე არხების ღერძზე (channel dimension)

ლექტორს უთხარი: “UNetUp არა მარტო upsample-ს აკეთებს, არამედ skip features-ს აერთებს, რომ დეტალები არ დაიკარგოს.”

5. Skip connections:
    Direct concatenation of encoder features into decoder
    Preserve fine spatial details (edges, textures)

   UNetSkipConnectionBlock — U-Net-ის მთავარი “მატრიოშკა”

ეს კლასი აგებს U-Net-ს რეკურსიულად/ჩაშენებულად.

ბლოკის შიდა ლოგიკა

Down ნაწილი: LeakyReLU -> Conv(stride=2) -> Norm

Submodule: შიგნით ჩაშენებული შემდეგი ბლოკი (უფრო ღრმა დონე)

Up ნაწილი: ReLU -> ConvTranspose(stride=2) -> Norm

3 შემთხვევა:

outermost=True

იღებს input-ს, აკეთებს down → submodule → up

ბოლოს Tanh() → output [-1,1] დიაპაზონში (შენ Normalize-ს ერგება)

innermost=True

ეს არის “ბოთლნეკი” — ყველაზე პატარა რეზოლუცია, ყველაზე ღრმა ფიჩერები.

შუალედური ბლოკები

ბოლოს აკეთებს:

return torch.cat([x, self.model(x)], 1)


ანუ input x-ს skip connection-ად ამატებს decoder-ის output-ს.

ლექტორს უთხარი: “UNetSkipConnectionBlock არის ერთი U-Net საფეხური: downsample → deeper block → upsample და ბოლოს skip-ით concat.”

7. Final activation:
     Tanh, producing outputs in [−1,1]

UNetGenerator — როგორ აწყობს მთლიან ქსელს

აქ შენ ქმნი ყველაზე ღრმა ბლოკს:

unet_block = UNetSkipConnectionBlock(ngf*8, ngf*8, innermost=True)


მერე ამაზე ზედ “აშენებ” ბლოკებს:

რამდენიმე ngf*8 შუალედური (num_downs-ით განსაზღვრული)

შემდეგ თანდათან ამცირებ არხებს:

ngf*4 -> ngf*8

ngf*2 -> ngf*4

ngf -> ngf*2
და ბოლოს outermost ბლოკი აკეთებს output-ს:

UNetSkipConnectionBlock(output_nc, ngf, outermost=True)


ამ კოდში მთავარი იდეა: U-Net-ის სიღრმე/რამდენჯერ downsample ხდება კონტროლდება num_downs-ით.

ერთი ხაზით შედარება ResNet-თან (თუ გკითხავენ)

ResNet generator: “ბირთვულად” ინახავს ინფორმაციას residual ბლოკებით (x + F(x))

U-Net generator: დეტალებს ინახავს skip connections-ით (encoder-ის early features პირდაპირ მიჰყავს decoder-ში)

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

------------------------------------------------------------------------------------------------------------

LSGAN vs Hinge Loss

Hinge Loss Result
FID: 89.65889200145094
MiFID: 103.301895


1) d_hinge_loss(real_logits, fake_logits)

ეს არის Discriminator-ის loss.

Discriminator-ს 2 დავალება აქვს:

✅ რეალურ სურათებზე

უნდა თქვას: “real არის real” → ანუ logits იყოს მაღალი (+)

ამიტომ loss_real არის:

loss_real = mean(relu(1 - real_logits))


➡️ თუ real_logits >= 1 → 1 - real_logits <= 0 → relu = 0
✅ ანუ Discriminator უკვე კარგად აკეთებს და loss აღარ ემატება

➡️ თუ real_logits < 1 → loss > 0
❌ ანუ D-ს არ ჰყოფნის confidence real-ზე და დაისჯება.

✅ fake სურათებზე

უნდა თქვას: “fake არის fake” → ანუ logits იყოს დაბალი (-)

ამიტომ loss_fake არის:

loss_fake = mean(relu(1 + fake_logits))


➡️ თუ fake_logits <= -1 → 1 + fake_logits <= 0 → relu = 0
✅ loss არ ემატება, fake კარგად ამოიცნო

➡️ თუ fake_logits > -1 → loss > 0
❌ ანუ Discriminator ვერ “ჩაგდო” fake საკმარისად დაბლა

საბოლოოდ:
return loss_real + loss_fake


ანუ Discriminator-ს ვაიძულებთ:

real_logits იყოს ≥ 1

fake_logits იყოს ≤ -1

ეს “1” და “-1” არის ე.წ. margin.

2) g_hinge_loss(fake_logits)

ეს არის Generator-ის loss.

Generator-ს უნდა რომ Discriminator მოტყუვდეს და fake სურათებზე თქვას “real”.

ანუ უნდა რომ fake_logits გახდეს რაც შეიძლება დიდი.

ხოდა loss არის:

return -mean(fake_logits)


➡️ Generator როცა ზრდის fake_logits-ს → mean(fake_logits) იზრდება
➡️ -mean(fake_logits) მცირდება
✅ ანუ loss მინიმუმისკენ მიდის

მოკლე დასკვნა (ლექტორთან რომ თქვა)

Discriminator: ცდილობს real → +1 ზე მაღლა, fake → -1 ზე დაბლა

Generator: ცდილობს fake logits რაც შეიძლება მაღალი გახადოს (რომ D მოატყუოს)
