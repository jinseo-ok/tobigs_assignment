import numpy as np
import math

class TwoLayerNet():
    """
    2 Layer Network를 만드려고 합니다.

    해당 네트워크는 아래의 구조를 따릅니다.

    input - Linear - ReLU - Linear - Softmax

    Softmax 결과는 입력 N개의 데이터에 대해 개별 클래스에 대한 확률입니다.
    """

    def __init__(self, X, input_size, hidden_size, output_size, std=1e-4):
         """
         네트워크에 필요한 가중치들을 initialization합니다.
         initialized by random values
         해당 가중치들은 self.params 라는 Dictionary에 담아둡니다.

         input_size: 데이터의 변수 개수 - D
         hidden_size: 히든 층의 H 개수 - H
         output_size: 클래스 개수 - C

         """

         self.params = {}
         self.params["W1"] = std * np.random.randn(input_size, hidden_size)
         self.params["b1"] = np.random.randn(hidden_size)
         self.params["W2"] = std * np.random.randn(hidden_size, output_size)
         self.params["b2"] = np.random.randn(output_size)

    def forward(self, X, y=None):

        """

        X: input 데이터 (N, D)
        y: 레이블 (N,)

        Linear - ReLU - Linear - Softmax - CrossEntropy Loss

        y가 주어지지 않으면 Softmax 결과 p와 Activation 결과 a를 return합니다. p와 a 모두 backward에서 미분할때 사용합니다.
        y가 주어지면 CrossEntropy Error를 return합니다.

        """
        # W1 == (D,H), b1 == (H,)
        W1, b1 = self.params["W1"], self.params["b1"]
        # W2 == (H,C), b2 == (C,)
        W2, b2 = self.params["W2"], self.params["b2"]
        N, D = X.shape # X == (N,D)

        # 여기에 p를 구하는 작업을 수행하세요.
        
        # Linear1 계층
        print(W1.shape)
        h = np.dot(X, W1) + b1  # H == (N,H)
        print('@@@')
        # ReLU 계층
        mask = (h <= 0)
        a = h.copy()
        a[mask] = 0 # a == (N,H)
        
        # Linear2 계층
        h2 = np.dot(a, W2) + b2 # h2 == (N,C)
        
        # softmax 계층
        '''
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        '''
        o = np.exp(h2 - np.max(h2))
        p = np.exp(o)/np.sum(np.exp(o),axis=1).reshape(-1,1) # p == (N,C)

        if y is None:
            return p, a
        
        # 여기에 Loss를 구하는 작업을 수행하세요.(y가 있는 상태)
        log_likelihood = 0
        i = 0
        for q in y:
            log_likelihood -= np.log(p[i,q])
            i += 1
            
        Loss = log_likelihood / N # N은 데이터 개수

        print('loss : ',Loss)

        return Loss



    def backward(self, X, y, learning_rate=1e-5):
        """

        X: input 데이터 (N, D)
        y: 레이블 (N,)

        grads에는 Loss에 대한 W1, b1, W2, b2 미분 값이 기록됩니다.

        원래 backw 미분 결과를 return 하지만
        여기서는 Gradient Descent방식으로 가중치 갱신까지 합니다.

        """
        print('adawasd')
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        N = X.shape[0] # 데이터 개수
        grads = {}
        print("진입")
        p, a = self.forward(X,y)

        # 여기에 파라미터에 대한 미분을 저장하세요.

        # softmax backpropagation
        # 요컨대 Softmax-with-Loss 노드의 역전파 그래디언트를 구하려면 입력값에 소프트맥스 확률값을 취한 뒤, 정답 레이블에 해당하는 요소만 1을 빼주면 된다는 얘기
        dp = p
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                if(j==y[i]):
                    dp[i][j]-=1 # p-y
        print('@@')
        # linear2 backpropagation
        grads["W2"] = np.dot(a.T, dp)
        print(1)
        grads["b2"] = np.sum(dp,axis=0)
        dl = np.dot(dp, W2.T)
        
        # ReLU backpropagation..?
        '''
                            0 if x1 < 0
        heaviside(x1, x2) = x2 if x1 == 0
                            1 if x1 > 0
        '''
        da = np.heaviside(a,0)

        # linear1 backpropagation
        dc = dl * da
        grads["W1"] = np.dot(X.T, dc)
        grads["b1"] = np.sum(dc, axis=0)

        self.params["W2"] -= learning_rate * grads["W2"]
        self.params["b2"] -= learning_rate * grads["b2"]
        self.params["W1"] -= learning_rate * grads["W1"]
        self.params["b1"] -= learning_rate * grads["b1"]

    def accuracy(self, X, y):

        p, _ = self.forward(X)
        
        
        pre_p = np.argmax(p,axis=1)

        return np.sum(pre_p==y)/pre_p.shape[0]
