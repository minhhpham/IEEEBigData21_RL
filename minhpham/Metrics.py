
class Metrics:
    def __init__(self, recommended_testset, purchaseLabels_testset, itemPrice):
        """ recommended_testset: list
            purchaseLabels_testset: list
            itemPrice: list
        """
        self.rec = recommended_testset
        self.labels = purchaseLabels_testset
        self.price = itemPrice
    def calculate_metrics1(self, recommendedItems):
        """
        recommendedItems: list of length equal to recommended_testset
        metrics calculated by summing total rewards of purchased items, no punishment
        """
        score = 0
        for i in range(len(recommendedItems)):
        # loop each sample in data
            predItems = recommendedItems[i]
            givenItems = self.rec[i]
            labels = self.labels[i]
            purchaseAND = [givenItems[i] for i in range(9) if labels[i]==1]
            for item in predItems:
            # loop each items in the sample
                if item in purchaseAND:
                    score = score + self.price[item-1]
        return score
                    
        
# def main():
    
if __name__ == '__main__':
    main()
    