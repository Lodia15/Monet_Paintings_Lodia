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
დავწერე: ResNetGenerator(nn.Module), PatchDiscriminator(nn.Module), training loop (manual).

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
  ვაპატარავებთ რეზოლუციას რომ network-მა უფრო გლობალური სტრუქტურა დაინახოს და იაფი გახდეს გამოთვლა.
  
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

ძირითადი პარამეტრები:
kw = 4 → ყველა convolution არის 4×4 kernel
padw = 1 → padding=1, რომ spatial ზომები კონტროლირებადად იცვლებოდეს
ndf = 32 → საწყისი feature რაოდენობა (სტანდარტულად ხშირად 64, მაგრამ 32 უფრო სწრაფია)

1)პირველი ფენა — “საწყისი დეტალების ამოღება + downsample”:

Conv2d(input_nc=3 → ndf=32, kernel=4, stride=2, padding=1)
LeakyReLU


რას აკეთებს:
იღებს RGB სურათს (3 არხი) და გარდაქმნის 32 feature map-ად
stride=2 ნიშნავს ზომის 2-ჯერ შემცირებას (downsample)
ანუ დისკრიმინატორი იწყებს კომპაქტურ წარმოდგენაზე მუშაობას.
LeakyReLU(0.2) გამოიყენება discriminator-ებში, რადგან “ჩვეულებრივ ReLU-ზე” უკეთ ინარჩუნებს gradient-ს და ხშირად სტაბილურობას აუმჯობესებს GAN training-ში

2)შუა ფენები (loop) — “Feature-ების გაღრმავება + არხების ზრდა”

Loop-ში ხდება ეს:
არხების გაზრდა სქემით: 32 → 64 → 128 (რადგან nf_mult = 2^n, მაქსიმუმ 8-მდე)
ყველა ასეთ ფენაში გაქვს:

Conv2d(..., stride=2, bias=False)
InstanceNorm2d
LeakyReLU

რატომ ასე:
stride=2 კვლავ ამცირებს spatial ზომას და ზრდის receptive field-ს → დისკრიმინატორი “უფრო დიდ კონტექსტს” ხედავს.
InstanceNorm2d ძალიან მნიშვნელოვანია CycleGAN-ში, განსაკუთრებით როცა batch_size=1 გაქვს:
BatchNorm ასეთ შემთხვევაში არასტაბილურია, ხოლო InstanceNorm თითო სურათს ცალკე ასტაბილურებს (სტილის ტრანსფერში კლასიკური არჩევანია).
bias=False იმიტომ, რომ InstanceNorm უკვე აკეთებს center/scale-ს მსგავს ეფექტს და bias ხშირად ზედმეტია.

3)“წინა-საბოლოო” ფენა — stride=1 (ზომას თითქმის აღარ აკლებს)
Conv2d(..., stride=1)
InstanceNorm2d
LeakyReLU

აქ იდეაა:
უკვე გვაქვს კარგი feature representation, და ახლა დისკრიმინატორი უფრო ზუსტ, ადგილობრივ შეფასებაზე გადადის ზომის ზედმეტი დაპატარავების გარეშე.

4)საბოლოო ფენა — 1 არხიანი score map
Conv2d(... → 1, stride=1)

ეს ფენა აბრუნებს (H×W×1) map-ს, სადაც თითო ელემენტი არის ლოგიტი/სქორი კონკრეტული patch-ისთვის.
ანუ output არ არის ერთი რიცხვი.
ეს არის “რეალურობის რუკა”.



# Loss Funcs

ვიყენებ სამნაირ loss-ს.
1. Adversarial loss(LSGAN)
   
   def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)


3. Cycle-consistency loss
   "აკონტროლებს გადასვლებს":
     P → M → P should reconstruct original P
     M → P → M should reconstruct original M
     

   def l1_loss(pred, target):
    return torch.mean(torch.abs(pred - target))


5. Identity loss
   ვცდილობთ არ დავუშვათ არასიჭირო ფერების ცვლა. 
   ჩვენ რომ მონეს სურათი გავუშვათ მონე->სურათი გენერატორში, ზუსტად იგივე უნდა დაგვიბრუნოს
     G_P2M(M) ≈ M, G_M2P(P) ≈ P
     


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



1)UNetGenerator — რას აკეთებს?

ეს კლასი აწყობს მთელ U-Net-ს “მატრიოშკას” პრინციპით:
ჯერ ქმნის ყველაზე ღრმა ბლოკს (bottleneck)
მერე მას “დააცვამს” ზედა ბლოკებს (უფრო და უფრო ზედაპირისკენ)
ბოლოს ქმნის outermost ბლოკს რომელიც აბრუნებს საბოლოო სურათს

unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
ეს არის U-Net-ის ყველაზე ღრმა ბლოკი:
აქ feature map არის ყველაზე პატარა ზომაზე
აქ მოდელი სწავლობს ყველაზე “გლობალურ” ინფორმაციას (style, composition)

for i in range(num_downs - 5):
    unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block)
აქ შენ აკეთებ შუა ბლოკებს:
არხები ისევ ngf*8 რჩება
უბრალოდ “სიღრმეს” უმატებ ქსელს

unet_block = UNetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block)
unet_block = UNetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block)
unet_block = UNetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block)

ეს ეტაპი არის encoder/decoder გადასვლის ბლოკები, სადაც
layer-ები ხდება უფრო “ზედაპირისკენ”
არხები ნელ-ნელა მცირდება (8→4→2→1)

self.model = UNetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True)

ეს არის outermost ბლოკი
იღებს input RGB-ს (Photo ან Monet)
და აბრუნებს output RGB-ს
ბოლო activation არის Tanh() → output [-1,1], რაც შენს Normalize-ს ერგება

2)UNetSkipConnectionBlock — ეს არის მთავარი ნაწილი

ეს კლასი არის 1 U-Net ბლოკი, რომელიც აკეთებს:

ძირითადი სტრუქტურა:
Down → (Inner block) → Up → Skip Connection

ანუ:
downsample (H,W ნახევრდება)
გადადის უფრო ღრმად (submodule)
upsample (H,W ორმაგდება)
skip: input feature-ს concat-ით ამატებს output-ზე

downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1)
მთავარი:
stride=2 → ზომას ორჯერ ამცირებს
input_nc -> inner_nc → არხებს ზრდის

upconv = nn.ConvTranspose2d(..., kernel_size=4, stride=2, padding=1)
მთავარი:
ConvTranspose2d → ზომას ორჯერ ზრდის
decoder აბრუნებს დეტალებს back to 256x256-ზე

3 შემთხვევა UNetSkipConnectionBlock-ში
1) outermost=True

ეს არის network-ის “კარი”:

down = [downconv]
up = [uprelu, upconv, nn.Tanh()]
model = down + [submodule] + up


ანუ outermost:

downsample-ს აკეთებს
გაუშვებს inner U-Net-ში
upsample
ბოლოს Tanh() იძლევა output-ს

outermost-ში არ არის concat დაბრუნებაზე, რადგან ეგ concat ქვედა ბლოკებში ხდება.

2)innermost=True
ეს არის bottleneck (ყველაზე ღრმა):

down = [downrelu, downconv]
up = [uprelu, upconv, upnorm]

აქ:
უკვე აღარ აქვს submodule (ესაა ბოლო)
უბრალოდ down + up

3)შუა ბლოკები (default)

ესაა ყველაზე მნიშვნელოვანი U-Net-ის სიძლიერე:

model = down + [submodule] + up

და forward-ში:
return torch.cat([x, self.model(x)], 1)


ეს არის Skip Connection:

x არის encoder feature map
self.model(x) არის decoder output
torch.cat(..., 1) აერთიანებს channel dimension-ზე

ანუ decoder იღებს encoder-ის დეტალებს პირდაპირ:
edges
textures
small patterns
ამიტომ U-Net უფრო კარგად ინახავს დეტალებს ვიდრე ResNet.

ResNet გენერატორი: “გადაკეთებას აკეთებს residual blocks-ით”
U-Net გენერატორი: “გადაკეთებას აკეთებს encoder-decoder-ით და დეტალებს აბრუნებს skip connections-ით”

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

---------------------------
def d_hinge_loss(real_logits, fake_logits):
    loss_real = torch.mean(torch.relu(1.0 - real_logits))
    loss_fake = torch.mean(torch.relu(1.0 + fake_logits))
    return loss_real + loss_fake

def g_hinge_loss(fake_logits):
    return -torch.mean(fake_logits)  #Hinge
---------------------------

აქ მთავარი სიტყვაა: logits.

real_logits = D(real_images)
fake_logits = D(fake_images)

ეს logits არის რიცხვები (PatchGAN-ში grid/map), რომლებიც არ არიან probability (0..1).
ანუ დისკრიმინატორი არ აკეთებს sigmoid-ს, უბრალოდ “score”-ებს აბრუნებს.

1.1 Discriminator hinge loss
Real სურათებისთვის D-მ უნდა დააბრუნოს score >= +1
Fake სურათებისთვის D-მ უნდა დააბრუნოს score <= -1
ანუ დისკრიმინატორი ცდილობს margin separation-ს:

Real → +1-ზე მაღლა
Fake → -1-ზე დაბლა

რას აკეთებს თითო ნაწილი?

Real ნაწილი
relu(1 - real_logits)

თუ real_logits >= 1 → 1-real_logits <=0 → ReLU = 0
ანუ loss = 0 (კაია, რეალს სწორად აფასებს)

თუ real_logits < 1 → loss დადებითია
ანუ დისკრიმინატორი ისჯება, რადგან რეალს “არ ყოფნის რეალურობა”

Fake ნაწილი
relu(1 + fake_logits)

თუ fake_logits <= -1 → 1+fake_logits <=0 → loss = 0
ანუ fake სწორადაა “გადაგდებული”

თუ fake_logits > -1 → loss იზრდება
fake ზედმეტად “რეალურად” ჩათვალა

1.2 Generator hinge loss
return -torch.mean(fake_logits)

Generator-ის მიზანია:
fake_logits რაც შეიძლება დიდი იყოს (positive / რეალზე მსგავსი)
რადგან:
თუ fake_logits დიდია → -mean(fake_logits) პატარაა → loss მცირდება 
ანუ Generator ცდილობს:
D(fake) → დიდი იყოს (რეალურობის score)


2) LSGAN loss (MSELoss) რას აკეთებს?

CycleGAN-ის LSGAN ვარიანტი გამოიყენებს:
D-სთვის: MSE(real, 1) და MSE(fake, 0)
G-სთვის: MSE(fake, 1)

ანუ:
Discriminator:
real → 1
fake → 0

Generator:
fake → 1
აქ D აკეთებს regression-ს, როგორც “რამდენად რეალურია 0-დან 1-მდე”.


3) მთავარი განსხვავებები: Hinge vs LSGAN
   განსხვავება #1: Margin vs Regression
Hinge:
დისკრიმინატორს აქვს margin rule:
real must be ≥ 1
fake must be ≤ -1
თუ უკვე margin დაკმაყოფილდა → loss=0 → აღარ “აწვება”.
ეს ხშირად იძლევა უფრო “ჯანსაღ” სწავლას.

LSGAN:
დისკრიმინატორი მუდმივად ცდილობს მიიყვანოს:
real → 1 ზუსტად
fake → 0 ზუსტად
ანუ უსასრულოდ “ზეწოლს” აგრძელებს, თუნდაც უკვე ძალიან კარგი იყოს.

განსხვავება #2: Gradient saturation / სწავლების სტაბილურობა
Hinge:
როცა D ძალიან ძლიერია და ყველაფერს სწორად ასხვავებს
მაინც ხშირად აქვს კარგი gradient-ები generator-ისთვის
ამიტომ ძალიან პოპულარულია GAN-ებში (მაგ StyleGAN და სხვ.)

LSGAN:
ხშირად უფრო smooth/stableა ვიდრე BCE, მაგრამ მაინც:
თუ D ძალიან confident გახდა → G-ს შეიძლება gradient “დაუჩუმდეს” ან დასუსტდეს.

განსხვავება #3: output scale
Hinge:
output logits შეიძლება იყოს ნებისმიერი რეალური რიცხვი
არ არის შეზღუდული [0,1]-ზე
D-სთვის უფრო მარტივია separation.

LSGAN:
მოდელი “პრობლემას უყურებს როგორც regression”-ს
შედეგი უფრო რბილი/ნაკლებად მკვეთრია ხშირად

განსხვავება #4: Generator-ის მოტივაცია
Hinge generator:
-mean(D(fake))
Generator ცდილობს უბრალოდ score-ის გაზრდას.

LSGAN generator:
MSE(D(fake), 1)
Generator ცდილობს მიუახლოვდეს კონკრეტულ მნიშვნელობას 1-ს.

Run summary:
D_M	0.20152
D_P	0.95512
G_M2P_adv	0.93634
G_P2M_adv	1.55101
G_total	6.24328
cycle_loss	2.48398
idt_M	0.72867
idt_P	0.54327

