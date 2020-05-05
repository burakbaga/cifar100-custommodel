import matplotlib.pyplot as plt
plt.style.use("seaborn-darkgrid")
def plot(train_accuracy,val_accuracy,train_loss,val_loss,model_name=""):
 
    epochs = range(1, len(train_accuracy)+1)

    plt.plot(epochs, train_accuracy, "s-", color="orange",label = "Train Accuracy")
    plt.plot(epochs, val_accuracy, "^-",color="red" ,label = "Validation Accuracy")
    plt.title("Train ve Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(f'grafikler/{model_name}_accuracy.png')
    plt.figure()


    plt.plot(epochs, train_loss,"s-",color="orange", label = "Train Loss")
    plt.plot(epochs, val_loss, "^-",color="red", label = "Validation Loss")
    plt.title("Train ve Validation Loss ")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(f'grafikler/{model_name}_loss.png')
    plt.show()