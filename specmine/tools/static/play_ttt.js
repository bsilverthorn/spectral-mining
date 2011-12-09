"use strict";

(function($) {

var assert = function(expression, message) {
    if(!expression) {
        throw { message: message };
    }
}

//
// UTILITY
//

var zeros = function(count) {
    var values = [];

    for(var i = 0; i < count; ++i) {
        values[i] = 0;
    }

    return values;
};

var manyZeros = function(rows, count) {
    var values = [];

    for(var i = 0; i < rows; ++i) {
        values[i] = zeros(count);
    }

    return values;
};

var normalized = function(values) {
    var total = values.reduce(function(a, b) { return a + b; });

    return values.map(function(v) { return v / total; });
};

var rescale = function(values) {
    var floor = values.reduce(function(a, b) { return Math.min(a, b); });
    var ceiling = values.reduce(function(a, b) { return Math.max(a, b); });

    return values.map(function(v) {
        if(v == null) {
            return 0.0;
        }
        else if(floor == ceiling) {
            return 0.5;
        }
        else {
            return (v - floor) / (ceiling - floor);
        }
    });
};

var softmax = function(values) {
    var temperature = 0.05;

    var exped = values.map(function(v) {
        if(v == null) {
            return 0.0;
        }
        else {
            return Math.exp(v / temperature);
        }
    });

    return normalized(exped);
};

var valuesToAction = function(values) {
    var probabilities = softmax(values);
    var r = Math.random();
    var total = 0;

    for(var i = 0; i < probabilities.length - 1; ++i) {
        var probability = probabilities[i];

        if(!isNaN(probability)) {
            total += probability;

            if(r < total) {
                return i;
            }
        }
    }

    return probabilities.length - 1;
};

//
// APPLICATION
//

var ttt = {
    player: 1,
    turn: 0,
    board: null
};

ttt.start = function() {
    this.board.start();
    this.modePlayer();
};

ttt.getWinner = function() {
    // check the rows and columns
    var rows = [0, 0, 0];
    var columns = [0, 0, 0];
    var filled = 0;

    for(var r = 0; r < 3; ++r) {
        for(var c = 0; c < 3; ++c) {
            var piece = this.board.state[r * 3 + c];

            rows[r] += piece;
            columns[c] += piece;

            filled += Math.abs(piece);
        }
    }

    for(var i = 0; i < 3; ++i) {
        if(rows[i] == 3 || columns[i] == 3) {
            return 1;
        }
        else if(rows[i] == -3 || columns[i] == -3) {
            return -1;
        }
    }

    // check the diagonals
    var rdiagonal = this.board.state[0] + this.board.state[4] + this.board.state[8];
    var ldiagonal = this.board.state[2] + this.board.state[4] + this.board.state[6];

    if(rdiagonal == 3 || ldiagonal == 3) {
        return 1;
    }
    else if(rdiagonal == 3 || ldiagonal == 3) {
        return -1;
    }

    // check for a draw
    if(filled == 9) {
        return 0;
    }

    // ...
    return null;
};

ttt.makeMove = function(index) {
    var self = this;

    // make the move
    self.board.play(index, self.player);

    self.player *= -1;
    self.turn += 1;

    // game over?
    var winner = self.getWinner();

    if(winner == null) {
        if(self.player == 1) {
            self.modePlayer();
        }
        else if(self.player == -1) {
            self.modeOpponent();
        }
        else {
            assert(false);
        }
    }
    else {
        self.modeWinner(winner);
    }
};

ttt.refreshPrompt = function(text) {
    d3.select("#prompt").html("&gt; " + text);
};

ttt.modeWinner = function(winner) {
    var self = this;

    if(winner == 0) {
        self.refreshPrompt("The only way to win... is not to play.");
    }
    else if(winner == 1) {
        self.refreshPrompt("Human wins. <a href=\"play_ttt.html\">Another game?</a>");
    }
    else if(winner == -1) {
        self.refreshPrompt("Human loses. <a href=\"play_ttt.html\">Another game?</a>");
    }

    self.board.active = false;
};

ttt.analyze = function(success) {
    $.ajax({
        url: "/analyze_board",
        data: {
            player: this.player,
            board: JSON.stringify(this.board.state)
        },
        success: success,
        error: function(data, textStatus, xhr) {
            var request = this;

            console.log("Request error! " + textStatus);

            window.setTimeout(function() { $.ajax(request); }, 500);
        }
    });
};

ttt.modePlayer = function() {
    var self = this;

    if(self.turn == 0) {
        self.refreshPrompt("Would you like to play a game?");
    }
    else {
        self.refreshPrompt("It's a strange game.");
    }

    $(this.board).one("moveRequest", function(event, i) {
        self.makeMove(i);
    });

    self.board.active = true;

    self.board.values = manyZeros(7, 9);
    self.board.hue = 200;

    self.board.refresh();

    self.analyze(function(data, textStatus, xhr) {
        self.board.values = data.values;
        self.board.numbers = data.numbers;
        self.board.weights = data.weights;

        self.board.refresh();
    });
};

ttt.modeOpponent = function() {
    var self = this;

    self.board.active = false;

    self.refreshPrompt("Your opponent is thinking.");

    self.board.values = manyZeros(7, 9);
    self.board.hue = 10;

    self.board.refresh();

    // request analysis for opponent
    self.analyze(function(data, textStatus, xhr) {
        self.board.values = data.values;
        self.board.numbers = data.numbers;
        self.board.weights = data.weights;

        self.board.refresh();

        window.setTimeout(
            function() {
                var index = valuesToAction(data.values[0]);

                self.makeMove(index);
            },
            2000);
    });
};

//
// BOARD
//

ttt.board = {
    state: zeros(9),
    values: manyZeros(7, 9),
    numbers: zeros(6),
    weights: zeros(6),
    hue: 200,
    active: true
};

ttt.board.start = function() {
    var self = this;

    this.refresh();

    d3.selectAll("div.square")
        .on("click", function(d, i) {
            if(self.active && d == 0) {
                $(self).trigger("moveRequest", i);
            }
        })
        .on("mouseover", function(d) {
            if(self.active && d == 0) {
                d3.select(this).text("X");
            }
        })
        .on("mouseout", function(d) {
            if(self.active && d == 0) {
                d3.select(this).text("");
            }
        });

    d3.select("#small-boards")
        .selectAll("div.small-board-area")
        .data([1, 2, 3, 4, 5, 6])
        .enter()
        .append("div")
        .classed("small-board-area", true)
        .selectAll("div.small-board")
        .data([0])
        .enter()
        .append("div")
        .classed("small-board", true)
        .selectAll("div.row")
        .data([0, 0, 0])
        .enter()
        .append("div")
        .classed("row", true)
        .selectAll("div.small-square")
        .data([0, 0, 0])
        .enter()
        .append("div")
        .classed("small-square", true);

    d3.selectAll("div.small-board-area")
        .selectAll("div.description")
        .data([0])
        .enter()
        .append("div")
        .classed("description", true);
};

ttt.board.play = function(index, player) {
    assert(this.state[index] == 0);

    this.state[index] = player;

    this.refresh();

    d3.selectAll("div.square")
        .filter(function(d, i) { return i == index; })
        .style("background-color", "#aaaaaa")
        .transition()
        .duration(750)
        .style("background-color", "#ffffff");
};

ttt.board.refresh = function() {
    var self = this;
    var probabilities = self.values.map(function(vf) { return softmax(vf) });

    var pieceToText = function(d) {
        if (d == 1) {
            return "X";
        }
        else if (d == -1) {
            return "O";
        }
        else if (d == 0) {
            return "";
        }
        else {
            assert(false);
        }
    };

    d3.selectAll("div.square")
        .data(this.state)
        .text(pieceToText)
        .classed("open", function(d) {
            return d == 0;
        })
        .transition()
        .duration(500)
        .style(
            "background-color",
            function(d, i) {
                var p = probabilities[0][i];

                return d3.hsl(self.hue, 0.6, 1.0 - p * 0.8);
            });

    var rescaled = [];

    for(var i = 1; i < 7; ++i) {
        rescaled[i - 1] = rescale(self.values[i]);
    }

    d3.selectAll("div.small-board")
        .data(rescaled)
        .selectAll("div.small-square")
        .data(function(d) { return d; })
        .text(function(d, i) { return pieceToText(self.state[i]); })
        .transition()
        .duration(500)
        .style(
            "background-color",
            function(d, i) {
                return d3.hsl(self.hue, 0.6, 1.0 - d * 0.8);
            });

    d3.selectAll("div.description")
        .data(self.numbers)
        .html(function(d, i) {
            return "N: " + self.numbers[i] + "<br>" + "W: " + self.weights[i].toFixed(2);
        });
};

//
// ...
//

$(window).load(function() {
    ttt.start();
});

})(jQuery.noConflict());

