#include "include\\actionList.h"
#include <iostream>
#include<iomanip>
#include "include\\chess.h"
#include "include\\autoPlayer.h"
#include "include\\humanPlayer.h"

using namespace std;

int main(){
    chess Game;
    Game.Players[1] = new humanPlayer("Umair", White);
    Game.Players[0] = new autoPlayer();
    Game.playGame();
    return 0;
}
