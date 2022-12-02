# Methods

## minimize the distance

在PTQ的校准过程中，我们认为矩阵乘法的原始输出$O=AB$和量化后的矩阵输出$\hat O = \Delta_A \Delta_B A_q B_q$越接近越好，换句话说：

$$
    \min_{\Delta_A,\Delta_B} dist(O, \hat O)
$$

那么如何选取最适合度量的$dist$呢？传统的做法是使用MSEloss和Simloss。本文通过实验表明传统的方法不能完美匹配模型的task loss。本文将量化后的模型等效于加入一个“微扰”，使用gradient和hessian来模拟损失。