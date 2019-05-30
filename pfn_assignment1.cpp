#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cstdlib>

using namespace std; 
using namespace Eigen;


//グラフクラスの定義
class Graph{
private:
    int dim_;//特徴量ベクトルの次元
    int v_num_;//頂点数
    MatrixXd G_;//行列形式でのグラフ表現，ij成分はノードiとノードjが繋がっているとき1，それ以外と対角成分は0
    MatrixXd W_;//パラメータ行列
    MatrixXd X_;//特徴量ベクトルを並べた行列
    VectorXd hg_;//グラフGに対する特徴量ベクトル
public:
    Graph(MatrixXd G,MatrixXd W);//コンストラクタ,引数は(G,W)
    MatrixXd concentration(int);//集約,引数は集約回数 
};



int main(){
    
    //グラフ，初期値設定
    Matrix3d G;//行列形式でのグラフ表現，ij成分はノードiとノードjが繋がっているとき1，それ以外と対角成分は0
    G << 0,1,1,
         1,0,0,
         1,0,0;
         

    Matrix2d W;//パラメータ行列 
    W << 0.5,1,
         0.5,-0.3;

    Graph SampleGraph(G,W);

    //T回の集約
    int T=2;
    MatrixXd hg = SampleGraph.concentration(T);//特徴量ベクトルの取得
    
    //特徴量ベクトルhgの出力
    cout << "hg=" << endl;
    cout << hg << endl; 
    
    return 0;
}

Graph::Graph(MatrixXd G,MatrixXd W):dim_(W.rows()),v_num_(G.rows()){
    if((G.transpose() != G) || (W.rows() != W.cols())){
        std::cout << "error!:Input Matrix Dimension is Maldefined! " << std::endl;
        std::exit(1);
    }//Matrixのサイズ確認
    G_=G;
    W_=W; 
    X_= MatrixXd::Ones(dim_,v_num_);//特徴量ベクトルの初期値は全ての要素が１とした
}

MatrixXd Graph::concentration(int T){

    MatrixXd Z=MatrixXd::Zero(dim_,v_num_);

    for(int i=0;i<T;i++){
        MatrixXd A=X_*G_;//集約1
        X_= (W_*A).cwiseMax(Z);//集約２
    }

    hg_=X_*VectorXd::Ones(v_num_);
    
    return hg_;
}




