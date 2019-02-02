#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h> 
#include <string.h>
#include "UDPCommunication.h"

void UDPCommInit (UDPCOMM *UDP, SETTINGS *set) {
  if (set->daqNum==1) {
    // captain code.

    // Creating socket file descriptor 

    if ( (UDP->sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) { 
      perror("socket creation failed"); 
      exit(EXIT_FAILURE); 
    } 
      
    memset(&UDP->servaddr, 0, sizeof(UDP->servaddr)); 
    memset(&UDP->cliaddr, 0, sizeof(UDP->cliaddr)); 
      
    // Filling server information 
    UDP->servaddr.sin_family    = AF_INET; // IPv4 
    UDP->servaddr.sin_addr.s_addr = inet_addr(set->sailor_bind); 
    UDP->servaddr.sin_port = htons(UDPCOMM_PORT); 

    struct timeval read_timeout;
    read_timeout.tv_sec = 0;
    read_timeout.tv_usec = 1;
    setsockopt(UDP->sockfd, SOL_SOCKET, SO_RCVTIMEO, &read_timeout, sizeof read_timeout);

 
    // Bind the socket with the server address 
    if ( bind(UDP->sockfd, (const struct sockaddr *)&(UDP->servaddr),  
	      sizeof(UDP->servaddr)) < 0 ) 
      { 
        perror("Bind failed"); 
        exit(EXIT_FAILURE); 
      } 
      
  } else if (set->daqNum==2) {
    // sailor code

    // Creating socket file descriptor 
    if ( (UDP->sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) { 
        perror("socket creation failed"); 
        exit(EXIT_FAILURE); 
    } 
  
    memset(&UDP->servaddr, 0, sizeof(UDP->servaddr)); 
      
    // Filling server information 
    UDP->servaddr.sin_family = AF_INET; 
    UDP->servaddr.sin_port = htons(UDPCOMM_PORT); 
    UDP->servaddr.sin_addr.s_addr = inet_addr(set->sailor_bind); 
      
  } else  {
    perror ("Bad daqNum!! Internal error. \n");
    exit(EXIT_FAILURE); 
    
  }
  
  

}

// BUFSIZE for messages;
#define BUFSIZE 2

void UDPPassKeyPress (UDPCOMM *UDP, char key) {
  char buffer[BUFSIZE];

  buffer[0]='~'; // meaning == keypress;
  buffer[1]=key;
  
  sendto(UDP->sockfd, (const char*)(&buffer), sizeof(BUFSIZE), 
	 MSG_CONFIRM, (const struct sockaddr *) &(UDP->servaddr),  
	 sizeof(&(UDP->servaddr))); 
}

void UDPGetKeyPress (UDPCOMM *UDP, char *key) {
  char buffer[BUFSIZE];
  int n;
  socklen_t len;
  
  n = recvfrom(UDP->sockfd, &buffer, BUFSIZE,  
	       MSG_WAITALL, ( struct sockaddr *) &UDP->cliaddr, 
	       &len); 

  if (n>0) {
    if (buffer[0]!='~') {
      printf ("Bad message received for captain\n");
      exit(EXIT_FAILURE); 
    }
    *key=buffer[1];
  };
}
