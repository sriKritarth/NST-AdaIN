import argparse
import torch
import torch.optim as optim
from pathlib import Path
from utils.utils import *
from utils.model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image



def parse_arguments():
   parser = argparse.ArgumentParser()

   parser.add_argument("--content_dataset" , type=str , default="test2017" , help="Image of content dataset")
   parser.add_argument("--style_dataset" , type=str , default="train_3" , help="Image of style dataset")
   parser.add_argument("--vgg" , type=str , default="vgg_normalised.pth" , help="Location of pretrained VGG")
   parser.add_argument("--experiment" , type=str , default="experiment1" , help="Name of experiment")

   parser.add_argument("--final_size" , type=int , default=256 , help="Final Size after transformation")
   parser.add_argument("--content" , type=int ,default=512 ,  help= "Size of content image")
   parser.add_argument("--style" ,  type=int ,default=512 ,  help= "Size of style image")

   parser.add_argument("--crop"  , action="store_true" , default=False , help="crop image")
   parser.add_argument("--batch_size" , type = int , default= 4 , help="Batch size of the image")

   parser.add_argument("--lr" , type=float , default=1e-4 , help="Learning rate")
   parser.add_argument("--lr_decay" , type=float , default = 5e-5 , help= "learning rate decay")
   parser.add_argument("--epoch" , type=int , default=1 , help="Training Epochs")
   parser.add_argument("--c_wt" , type=float , default= 1.0 , help="Content Weight")
   parser.add_argument("--s_wt", type=float , default=5.0 , help="Style weights")

   parser.add_argument("--log_interval" , type = int , default= 1 , help="Log interval")
   parser.add_argument("--save_interval" , type=int , default = 2 , help="Save interval")

   parser.add_argument("--resume" , action="store_true" , default=False , help="Resume training")
   parser.add_argument("--decoder_path" , type = str , default=None , help = "Path to decoder checkpoints")
   parser.add_argument("--optimizer_path" , type=str , default=None , help = "path to optimizer chekpoint")


   return parser.parse_args()



def main():
   args = parse_arguments()
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   save_dir = Path("experiment") / args.experiment
   save_dir.mkdir(exist_ok = True , parents = True)

   #save argument values
   with open(save_dir / 'args.txt' , 'w') as args_file:
      for keys , values in vars(args).items():
         args_file.write(f"{keys} : {values}\n")

   
   content_transform = get_transform(args.content , args.final_size , args.crop)
   style_transform = get_transform(args.style , args.final_size , args.crop)
   content_dataset = CustomImageDataset(args.content_dataset , transform=content_transform)
   style_dataset = CustomImageDataset(args.style_dataset , transform=style_transform)

   content_loader = DataLoader(content_dataset , batch_size=args.batch_size , shuffle=True , pin_memory=True , drop_last=True)
   style_loader = DataLoader(style_dataset , batch_size=args.batch_size , shuffle=True , pin_memory=True , drop_last=True )
   

   encoder = VggEncoder(args.vgg).to(device)
   decoder = Decoder().to(device)

   optimizer = optim.Adam(decoder.parameters() , lr = args.lr)

   scheduler = optim.lr_scheduler.LambdaLR(
      optimizer = optimizer,
      lr_lambda = lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch)
   )

   #Resume checkpoints
   if args.resume:
      decoder.load_state_dict(torch.load(args.decoder_path))
      optimizer.load_state_dict(torch.load(args.optimizer_path))

   
   print("Number of batches content-dataset" , len(content_loader))
   print("Number of batches style-dataset" , len(style_loader))

   print("Training...")

   criterion = nn.MSELoss()
   running_loss = None
   running_closs = None
   running_sloss = None

   encoder.eval()
   for epoch in range(args.epoch):
      decoder.train()
      progress_bar = tqdm(zip(content_loader , style_loader) , total= min(len(content_loader) , len(style_loader)))

      running_loss = 0
      running_closs = 0
      running_sloss = 0
      for c_image , s_image in progress_bar:
         
         optimizer.zero_grad()

         c_image = c_image.to(device , dtype = torch.float32 , non_blocking = True)
         s_image = s_image.to(device , dtype = torch.float32 , non_blocking = True)


         c_feats = encoder(c_image)
         s_feats = encoder(s_image)
         
         
         t = adaptive_instance_normalization(c_feats[-1] , s_feats[-1])
         g = decoder(t)
         g_feats = encoder(g)

         loss_content = criterion(g_feats[-1] , t) * args.c_wt

         loss_s = 0

         for g_f , s_f in zip(g_feats , s_feats):
            g_mean , g_std = calc_std_mean(g_f)
            s_mean , s_std = calc_std_mean(s_f)
            loss_s += criterion(g_mean , s_mean) + criterion(g_std , s_std) 

         loss_s = loss_s * args.s_wt

         loss = loss_content + loss_s
         loss.backward()
         optimizer.step()

         progress_bar.set_description(f"Loss : {loss.item() : 4f} , content-loss : {loss_content.item():4f} , style-loss : {loss_s.item():4f}")


         running_closs += loss_content.item()
         running_sloss += loss_s.item()
         running_loss += loss.item()

      scheduler.step()
      running_sloss /= len(content_loader)
      running_loss /= len(content_loader)
      running_closs /= len(content_loader)

      if(epoch + 1 ) % args.log_interval == 0:
         tqdm.write(f"iter = {epoch + 1} , loss = {running_loss : 4f} , content-loss = {running_closs : 4f} , style-loss  = {running_sloss : 4f}")
      
      #Model checkpoints
      if(epoch + 1) % args.save_interval == 0:
         torch.save(decoder.state_dict() , save_dir / f'decoder{epoch+1}.pth')
         torch.save(optimizer.state_dict() , save_dir / f'optimizer{epoch+1}.pth')

         with torch.no_grad():
            output = torch.cat([c_image , s_image , g])
            save_image(output , save_dir / f'output_{epoch+1}.png' , nrow =args.batch_size)



      

if __name__ == "__main__":
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(device)
   main()