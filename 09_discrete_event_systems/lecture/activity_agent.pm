mdp

module activity_agent
    // Information state
    // weather: 0=unknown, 1=known
    // mood: 0=unknown, 1=known
    weather : [0..1] init 0;
    mood : [0..1] init 0;
    
    // Interaction state
    // 0=active (still interacting)
    // 1=success (user accepted recommendation)
    // 2=abandoned (user left)
    status : [0..2] init 0;
    
    // === Actions when no information collected (state s_0) ===
    // Ask about weather first
    [ask_weather] (weather=0 & mood=0 & status=0) -> 
        0.75 : (weather'=1) +   // user provides weather -> s_W
        0.15 : true +           // unclear response -> stay in s_0
        0.10 : (status'=2);     // user abandons -> s_abandon
    
    // Ask about mood first  
    [ask_mood] (weather=0 & mood=0 & status=0) ->
        0.70 : (mood'=1) +      // user provides mood -> s_M
        0.20 : true +           // unclear response -> stay in s_0
        0.10 : (status'=2);     // user abandons -> s_abandon
    
    // Ask both at once (might overwhelm user)
    [ask_both] (weather=0 & mood=0 & status=0) ->
        0.50 : (weather'=1) & (mood'=1) +  // both answered -> s_WM
        0.20 : (weather'=1) +               // only weather -> s_W
        0.15 : (mood'=1) +                  // only mood -> s_M
        0.15 : (status'=2);                 // overwhelmed, abandons
    
    // === Actions when only weather is known (state s_W) ===
    [ask_mood] (weather=1 & mood=0 & status=0) ->
        0.80 : (mood'=1) +      // user provides mood -> s_WM
        0.10 : true +           // unclear response -> stay in s_W
        0.10 : (status'=2);     // user abandons
    
    // === Actions when only mood is known (state s_M) ===
    [ask_weather] (weather=0 & mood=1 & status=0) ->
        0.80 : (weather'=1) +   // user provides weather -> s_WM
        0.10 : true +           // unclear response -> stay in s_M
        0.10 : (status'=2);     // user abandons
    
    // === Action when both known (state s_WM): make recommendation ===
    [recommend] (weather=1 & mood=1 & status=0) ->
        0.75 : (status'=1) +    // user accepts -> s_success
        0.25 : (status'=2);     // user rejects and leaves -> s_abandon

    // Self-loop in success state to collect reward
    [] (status=1) -> true;

endmodule

// Labels for states
label "s_0" = (weather=0 & mood=0 & status=0);
label "s_W" = (weather=1 & mood=0 & status=0);
label "s_M" = (weather=0 & mood=1 & status=0);
label "s_WM" = (weather=1 & mood=1 & status=0);
label "s_success" = (status=1);
label "s_abandon" = (status=2);

// Rewards
rewards "task_completion"
    status=1 : 10;  // reward for being in success state
endrewards