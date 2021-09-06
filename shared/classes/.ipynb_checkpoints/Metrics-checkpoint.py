from classes.Items import Items

class Metrics:
    def __init__(self, recommendedItems, purchaseLabels):
        """ recommendedItems: list, the ground truth
            purchaseLabels: list, the ground truth
        """
        self.rec = recommendedItems
        self.labels = purchaseLabels
        self.itemsInfo = Items()

    def calculate_metrics1(self, recommendedItems_pred):
        """
        recommendedItems_pred: list of length equal to recommendedItems, these are predicted from models
        metrics calculated by summing total rewards of purchased items, no punishment
        """
        score = 0
        for i in range(len(recommendedItems_pred)):
        # loop each sample in data
            predItems = recommendedItems_pred[i]
            givenItems = self.rec[i]
            labels = self.labels[i]
            purchaseAND = [givenItems[i] for i in range(9) if labels[i]==1]
            for item in predItems:
            # loop each items in the sample
                if item in purchaseAND:
                    score = score + self.itemsInfo.getItemPrice(item)
        return score

    def calculate_metrics2(self, recommendedItems_pred, w1 = 1, w2 = 10, w3 = 100):
        """
        recommendedItems_pred: list of length equal to recommendedItems, these are predicted from models
        metrics calculated by summing total rewards of purchased items, no punishment
        w1, w2, w3: weights according to item's location
        metrics2 = sum of (purchasedLabel * itemPrice * (w1 or w2 or w3))
        """
        score = 0
        for i in range(len(recommendedItems_pred)):
        # loop each sample in data
            predItems = recommendedItems_pred[i]
            givenItems = self.rec[i]
            labels = self.labels[i]
            purchasedItems = [givenItems[i] for i in range(9) if labels[i]==1]
            for item in predItems:
                if item in purchasedItems:
                    itemLocation = self.itemsInfo.getItemLocation(item)
                    if itemLocation==1:
                        w = w1
                    elif itemLocation==2:
                        w = w2
                    elif itemLocation==3:
                        w = w3
                    score = score + self.itemsInfo.getItemPrice(item)*w
        return score
def main():
    return

if __name__ == '__main__':
    main()
    